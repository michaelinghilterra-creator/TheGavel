import asyncio
import copy
import string
from pathlib import Path
from types import MethodType

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from gavel.backends.native import NativeBackend, gavel_sdpa, gavel_sdpa_mask
from gavel.judge import build_user_message, judge
from gavel.rubric import load_rubric


FIXTURE = Path(__file__).parent / "fixtures" / "demo_rubric.yaml"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def native_factory():
    characters = list(dict.fromkeys(string.printable + "\n"))
    vocabulary = {character: index for index, character in enumerate(characters)}
    vocabulary["<eos>"] = len(vocabulary)
    vocabulary["<unk>"] = len(vocabulary)
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    raw.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}<u>{{ message['content'] }}</u>"
        "{% else %}<a>{{ message['content'] }}"
        "{% if not (loop.last and continue_final_message) %}</a>{% endif %}"
        "{% endif %}{% endfor %}"
    )

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=2048,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation="gavel_sdpa",
    )
    original = LlamaForCausalLM(config).eval()
    state = copy.deepcopy(original.state_dict())

    def make(**kw):
        model = LlamaForCausalLM(copy.deepcopy(config)).eval()
        model.load_state_dict(state)
        return NativeBackend(model, tokenizer, chunk=16, **kw)

    return make


def test_cached_logprobs_are_bit_identical_to_cold(native_factory) -> None:
    warm = native_factory()
    cold = native_factory()
    run(warm.next_token_logprobs("A" * 80, "prefix"))
    cached, _ = run(warm.next_token_logprobs("A" * 80 + " tail", "prefix"))
    direct, _ = run(cold.next_token_logprobs("A" * 80 + " tail", "prefix"))
    assert cached == direct


def test_gavel_sdpa_matches_default_sdpa_for_prefill_and_decode() -> None:
    torch.manual_seed(1)
    common = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
    )
    reference = LlamaForCausalLM(
        LlamaConfig(**common, attn_implementation="sdpa")
    ).eval()
    candidate = LlamaForCausalLM(
        LlamaConfig(**common, attn_implementation="gavel_sdpa")
    ).eval()
    candidate.load_state_dict(reference.state_dict())
    token_ids = torch.arange(50).remainder(common["vocab_size"]).unsqueeze(0)
    reference_cache = DynamicCache()
    candidate_cache = DynamicCache()

    with torch.inference_mode():
        for start in range(0, 50, 16):
            chunk = token_ids[:, start : start + 16]
            expected = reference(
                input_ids=chunk,
                past_key_values=reference_cache,
                use_cache=True,
            )
            actual = candidate(
                input_ids=chunk,
                past_key_values=candidate_cache,
                use_cache=True,
            )
            assert torch.allclose(actual.logits, expected.logits, atol=1e-5)
        for step in range(5):
            chunk = torch.tensor([[50 + step]])
            expected = reference(
                input_ids=chunk,
                past_key_values=reference_cache,
                use_cache=True,
            )
            actual = candidate(
                input_ids=chunk,
                past_key_values=candidate_cache,
                use_cache=True,
            )
            assert torch.allclose(actual.logits, expected.logits, atol=1e-5)


def test_gavel_sdpa_refuses_batch_size_two() -> None:
    query = torch.zeros(2, 4, 3, 8)
    key = torch.zeros(2, 2, 3, 8)
    value = torch.zeros(2, 2, 3, 8)
    with pytest.raises(ValueError, match="batch size 1"):
        gavel_sdpa(None, query, key, value, None)


def test_gavel_sdpa_mask_is_registered_and_skips_allocation() -> None:
    mask_function = ALL_MASK_ATTENTION_FUNCTIONS["gavel_sdpa"]
    assert mask_function is gavel_sdpa_mask
    assert mask_function(unused=True) is None


def test_static_cache_matches_dynamic_prefill_decode_and_judge(
    native_factory,
) -> None:
    static = native_factory()
    dynamic = native_factory()
    dynamic._new_cache = MethodType(lambda self: None, dynamic)
    token_ids = list(range(60))

    with torch.inference_mode():
        static_logits, static_cache, _ = static._prefill(token_ids)
        dynamic_logits, dynamic_cache, _ = dynamic._prefill(token_ids)
        assert torch.equal(static_logits, dynamic_logits)
        for _ in range(8):
            token_id = static._ordered_ids(static_logits[0], 1)[0]
            inputs = torch.tensor([[token_id]])
            static_output = static.model(
                input_ids=inputs,
                past_key_values=static_cache,
                use_cache=True,
            )
            dynamic_output = dynamic.model(
                input_ids=inputs,
                past_key_values=dynamic_cache,
                use_cache=True,
            )
            static_cache = static_output.past_key_values
            dynamic_cache = dynamic_output.past_key_values
            static_logits = static_output.logits[:, -1, :]
            dynamic_logits = dynamic_output.logits[:, -1, :]
            assert torch.equal(static_logits, dynamic_logits)

    rubric = load_rubric(FIXTURE)
    items = [
        {"id": "one", "text": "A clear short answer."},
        {"id": "two", "text": "A vague answer."},
        {"id": "three", "text": "A careful factual answer."},
    ]
    expected, _ = run(judge(dynamic, rubric, items, mode="lean", concurrency=1))
    actual, _ = run(judge(static, rubric, items, mode="lean", concurrency=1))
    assert actual == expected


def test_static_cache_crop_then_extend_matches_cold(native_factory) -> None:
    reused = native_factory()
    cold = native_factory()
    token_ids = list(range(64))

    with torch.inference_mode():
        _, cache, _ = reused._prefill(token_ids[:48])
        cache.crop(-16)
        for start in range(32, 64, 16):
            output = reused.model(
                input_ids=torch.tensor([token_ids[start : start + 16]]),
                past_key_values=cache,
                use_cache=True,
            )
        expected, _, _ = cold._prefill(token_ids)

    assert torch.equal(output.logits[:, -1, :], expected)


def test_static_cache_deepcopy_is_independent(native_factory) -> None:
    backend = native_factory()
    with torch.inference_mode():
        _, original, _ = backend._prefill(list(range(32)))
        original_length = original.get_seq_length()
        original_keys = original.layers[0].keys.clone()
        copied = copy.deepcopy(original)
        backend.model(
            input_ids=torch.tensor([[32]]),
            past_key_values=copied,
            use_cache=True,
        )

    assert copied.get_seq_length() == original_length + 1
    assert original.get_seq_length() == original_length
    assert torch.equal(original.layers[0].keys, original_keys)


def test_static_cache_model_forwards_do_not_sync_lengths(
    native_factory, monkeypatch
) -> None:
    backend = native_factory()
    cache = backend._new_cache()
    original_item = torch.Tensor.item
    calls = 0

    def counting_item(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_item(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", counting_item)
        with torch.inference_mode():
            output = backend.model(
                input_ids=torch.tensor([list(range(16))]),
                past_key_values=cache,
                use_cache=True,
            )
        assert calls == 0

        token_id = backend._ordered_ids(output.logits[:, -1, :][0], 1)[0]
        with torch.inference_mode():
            backend.model(
                input_ids=torch.tensor([[token_id]]),
                past_key_values=cache,
                use_cache=True,
            )
        assert calls <= 1


def test_static_cache_refuses_updates_past_max_len(native_factory) -> None:
    backend = native_factory(max_len=16)
    with pytest.raises(ValueError, match=r"17.*16"):
        backend._prefill(list(range(17)))


def test_pure_extensions_make_no_cache_copies(native_factory) -> None:
    warm = native_factory()
    cold = native_factory()
    prefixes = ["A" * 40, "A" * 80, "A" * 120]
    for prefix in prefixes:
        warm_result, _ = run(warm.next_token_logprobs("user", prefix))
    cold_result, _ = run(cold.next_token_logprobs("user", prefixes[-1]))
    assert warm.stats.copies == 0
    assert warm_result == cold_result


def test_judge_matches_copy_always_cache_behavior(native_factory) -> None:
    rubric = load_rubric(FIXTURE)
    items = [
        {"id": "one", "text": "A clear short answer."},
        {"id": "two", "text": "A vague answer."},
        {"id": "three", "text": "A careful factual answer."},
    ]
    current = native_factory()
    copy_always = native_factory()

    def copied_start(self, ids):
        best_key = None
        best_common = 0
        for key in self._cache:
            common = self._common_prefix(key, ids)
            if common > best_common:
                best_key = key
                best_common = common
        reused = (best_common // self.chunk) * self.chunk
        if best_key is None or reused == 0:
            return None, 0, None, {}
        cached = copy.deepcopy(self._cache[best_key])
        self.stats.copies += 1
        current_length = cached.get_seq_length()
        if current_length > reused:
            cached.crop(reused - current_length)
        known_logits = self._cache_logits[best_key]
        logits = known_logits.get(reused)
        boundaries = {
            length: value.clone()
            for length, value in known_logits.items()
            if length <= reused
        }
        self._cache.move_to_end(best_key)
        return (
            cached,
            reused,
            logits.clone() if logits is not None else None,
            boundaries,
        )

    copy_always._cached_start = MethodType(copied_start, copy_always)
    expected, _ = run(
        judge(copy_always, rubric, items, mode="lean", concurrency=1)
    )
    actual, _ = run(judge(current, rubric, items, mode="lean", concurrency=1))
    assert actual == expected


def test_judge_is_bit_identical_across_concurrency(native_factory) -> None:
    rubric = load_rubric(FIXTURE)
    items = [
        {"id": "one", "text": "A clear short answer."},
        {"id": "two", "text": "A vague answer."},
        {"id": "three", "text": "A careful factual answer."},
    ]
    sequential, _ = run(
        judge(native_factory(), rubric, items, mode="lean", concurrency=1)
    )
    concurrent, _ = run(
        judge(native_factory(), rubric, items, mode="lean", concurrency=3)
    )
    assert sequential == concurrent


def test_judging_shared_preamble_reuses_aligned_tokens(native_factory) -> None:
    rubric = load_rubric(FIXTURE)
    backend = native_factory()
    items = [
        {"id": "one", "text": "First item."},
        {"id": "two", "text": "Second item."},
    ]
    run(judge(backend, rubric, items, mode="lean", concurrency=1))
    preamble_ids = backend._prompt_ids(build_user_message(rubric, ""), "")
    aligned = (len(preamble_ids) // backend.chunk) * backend.chunk
    assert backend.stats.reused_tokens > 0
    assert backend.stats.reused_tokens >= aligned


def test_generation_honors_stop_max_tokens_and_eos(native_factory) -> None:
    backend = native_factory()
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    repeated = backend.tokenizer.decode([0])

    text, count = run(backend.generate("user", "", 8, [repeated * 2]))
    assert text == ""
    assert count == 2

    text, count = run(backend.generate("other", "", 3, []))
    assert text == repeated * 3
    assert count == 3

    old_eos = backend.tokenizer.eos_token_id
    backend.tokenizer.eos_token_id = 0
    try:
        text, count = run(backend.generate("eos", "", 3, []))
    finally:
        backend.tokenizer.eos_token_id = old_eos
    assert text == ""
    assert count == 1


def test_generation_stop_spanning_two_tokens_keeps_preceding_text(
    native_factory,
) -> None:
    backend = native_factory()
    token_ids = iter(backend.tokenizer.convert_tokens_to_ids(list("abc")))
    backend._ordered_ids = MethodType(
        lambda self, logprobs, count: [next(token_ids)], backend
    )

    text, count = run(backend.generate("user", "", 5, ["bc"]))

    assert text == "a"
    assert count == 3


def test_hybrid_attention_is_refused(native_factory) -> None:
    backend = native_factory()
    backend.model.config.layer_types = ["full_attention", "linear_attention"]
    with pytest.raises(ValueError, match="linear_attention"):
        NativeBackend(backend.model, backend.tokenizer, chunk=16)


def test_shorter_divergent_prompt_copies_and_preserves_entry(native_factory) -> None:
    backend = native_factory()
    run(backend.next_token_logprobs("Z" * 100, "prefix"))
    entry = next(iter(backend._cache.values()))
    original_length = entry.get_seq_length()
    run(backend.next_token_logprobs("Z" * 40, "prefix"))
    assert backend.stats.copies == 1
    assert entry.get_seq_length() == original_length


def test_top_logprobs_are_ordered_and_limited(native_factory) -> None:
    values, _ = run(native_factory().next_token_logprobs("user", "", top=7))
    assert len(values) == 7
    assert [item["logprob"] for item in values] == sorted(
        (item["logprob"] for item in values), reverse=True
    )
