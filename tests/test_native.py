import asyncio
import copy
import string
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gavel.backends.native import NativeBackend
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


def test_hybrid_attention_is_refused(native_factory) -> None:
    backend = native_factory()
    backend.model.config.layer_types = ["full_attention", "linear_attention"]
    with pytest.raises(ValueError, match="linear_attention"):
        NativeBackend(backend.model, backend.tokenizer, chunk=16)


def test_serving_shorter_prompt_does_not_mutate_entry(native_factory) -> None:
    backend = native_factory()
    run(backend.next_token_logprobs("Z" * 100, "prefix"))
    entry = next(iter(backend._cache.values()))
    original_length = entry.get_seq_length()
    run(backend.next_token_logprobs("Z" * 40, "prefix"))
    assert entry.get_seq_length() == original_length


def test_top_logprobs_are_ordered_and_limited(native_factory) -> None:
    values, _ = run(native_factory().next_token_logprobs("user", "", top=7))
    assert len(values) == 7
    assert [item["logprob"] for item in values] == sorted(
        (item["logprob"] for item in values), reverse=True
    )
