"""In-process Hugging Face causal language model backend."""

from __future__ import annotations

import asyncio
import copy
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

torch.use_deterministic_algorithms(True)


def gavel_sdpa(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    """Run causal SDPA without expanding grouped key and value heads."""

    if query.shape[0] != 1:
        raise ValueError("gavel_sdpa requires batch size 1")
    query_length = query.shape[2]
    key_length = key.shape[2]
    # Ignore HF's attention mask: batch size 1 has no padding, only causality.
    mask = (
        None
        if query_length == 1
        else causal_lower_right(query_length, key_length)
    )
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=dropout,
        scale=scaling,
        enable_gqa=key.shape[1] != query.shape[1],
    )
    return output.transpose(1, 2).contiguous(), None


AttentionInterface.register("gavel_sdpa", gavel_sdpa)


@dataclass
class NativeStats:
    """Timing and token counters for native inference."""

    prefill_tokens: int = 0
    reused_tokens: int = 0
    prefill_s: float = 0.0
    decode_tokens: int = 0
    decode_s: float = 0.0
    calls: int = 0
    copies: int = 0


class NativeBackend:
    """Run a causal language model locally with an aligned prefix cache."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: str = "cpu",
        chunk: int = 256,
        cache_entries: int = 3,
        top_default: int = 20,
    ):
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        if cache_entries < 0:
            raise ValueError("cache_entries must not be negative")

        layer_types = getattr(model.config, "layer_types", None) or ()
        offending = [kind for kind in layer_types if kind != "full_attention"]
        if getattr(model.config, "use_sliding_window", False):
            offending.append("sliding_window")
        if offending:
            names = ", ".join(dict.fromkeys(offending))
            raise ValueError(
                f"hybrid or sliding attention is not supported: {names}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chunk = chunk
        self.cache_entries = cache_entries
        self.top_default = top_default
        self.stats = NativeStats()
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[tuple[int, ...], object] = OrderedDict()
        self._cache_logits: dict[tuple[int, ...], dict[int, torch.Tensor]] = {}

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kw,
    ) -> "NativeBackend":
        """Load a local model and tokenizer without changing other backends."""

        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            attn_implementation="gavel_sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.eval()
        model.to(device)
        return cls(model, tokenizer, device=device, **kw)

    def _prompt_ids(self, user: str, prefix: str) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefix},
            ],
            continue_final_message=True,
            tokenize=True,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        while ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("chat template returned more than one prompt")
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    @staticmethod
    def _common_prefix(left: tuple[int, ...], right: list[int]) -> int:
        limit = min(len(left), len(right))
        for index in range(limit):
            if left[index] != right[index]:
                return index
        return limit

    def _cached_start(self, ids: list[int]):
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

        entry = self._cache[best_key]
        current = entry.get_seq_length()
        known_logits = self._cache_logits[best_key]
        if current == reused:
            cached = self._cache.pop(best_key)
            del self._cache_logits[best_key]
        else:
            cached = copy.deepcopy(entry)
            self.stats.copies += 1
        if current > reused:
            cached.crop(reused - current)
        logits = known_logits.get(reused)
        boundaries = {
            length: value.clone()
            for length, value in known_logits.items()
            if length <= reused
        }
        if best_key in self._cache:
            self._cache.move_to_end(best_key)
        return cached, reused, logits.clone() if logits is not None else None, boundaries

    def _store_cache(
        self, ids: list[int], cache, boundaries: dict[int, torch.Tensor]
    ) -> None:
        aligned = (len(ids) // self.chunk) * self.chunk
        if aligned == 0 or self.cache_entries == 0:
            return

        current = cache.get_seq_length()
        if current > aligned:
            cache.crop(aligned - current)
        key = tuple(ids[:aligned])

        for old_key in list(self._cache):
            if len(old_key) <= len(key) and key[: len(old_key)] == old_key:
                del self._cache[old_key]
                del self._cache_logits[old_key]
        self._cache[key] = cache
        self._cache_logits[key] = {
            length: value.clone()
            for length, value in boundaries.items()
            if length <= aligned
        }
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_entries:
            evicted, _ = self._cache.popitem(last=False)
            del self._cache_logits[evicted]

    def _prefill(self, ids: list[int]):
        if not ids:
            raise ValueError("chat template returned an empty prompt")

        started = time.perf_counter()
        cache, reused, logits, boundaries = self._cached_start(ids)
        for start in range(reused, len(ids), self.chunk):
            token_ids = ids[start : start + self.chunk]
            inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=inputs,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            end = start + len(token_ids)
            if end % self.chunk == 0:
                boundaries[end] = logits.clone()

        if logits is None:
            raise RuntimeError("aligned cache entry has no boundary logits")

        self.stats.reused_tokens += reused
        self.stats.prefill_tokens += len(ids) - reused
        self.stats.prefill_s += time.perf_counter() - started
        return logits, cache, boundaries

    @staticmethod
    def _ordered_ids(logprobs: torch.Tensor, count: int) -> list[int]:
        count = max(0, min(count, logprobs.numel()))
        ordered = torch.argsort(logprobs, descending=True, stable=True)
        return ordered[:count].tolist()

    @torch.inference_mode()
    def _next_token_logprobs_sync(
        self, user: str, prefix: str, top: int
    ) -> tuple[list[dict], int]:
        self.stats.calls += 1
        ids = self._prompt_ids(user, prefix)
        logits, cache, boundaries = self._prefill(ids)
        self._store_cache(ids, cache, boundaries)
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        values = [
            {
                "token": self.tokenizer.decode([token_id]),
                "logprob": float(logprobs[token_id].item()),
            }
            for token_id in self._ordered_ids(logprobs, top)
        ]
        return values, 1

    async def next_token_logprobs(
        self, user: str, prefix: str, top: int = 20
    ) -> tuple[list[dict], int]:
        """Return deterministic next-token log probabilities."""

        async with self._lock:
            return await asyncio.to_thread(
                self._next_token_logprobs_sync, user, prefix, top
            )

    @torch.inference_mode()
    def _generate_sync(
        self,
        user: str,
        prefix: str,
        max_tokens: int,
        stop: list[str],
    ) -> tuple[str, int]:
        self.stats.calls += 1
        ids = self._prompt_ids(user, prefix)
        logits, cache, boundaries = self._prefill(ids)
        generated: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        text = ""
        decode_started = time.perf_counter()

        for _ in range(max(0, max_tokens)):
            token_id = self._ordered_ids(logits[0].float(), 1)[0]
            generated.append(token_id)
            if token_id == eos_id:
                break

            text = self.tokenizer.decode(generated)
            found = [text.find(value) for value in stop if value and value in text]
            if found:
                text = text[: min(found)]
                break

            inputs = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=inputs,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        self.stats.decode_tokens += len(generated)
        self.stats.decode_s += time.perf_counter() - decode_started
        self._store_cache(ids, cache, boundaries)
        return text, len(generated)

    async def generate(
        self,
        user: str,
        prefix: str,
        max_tokens: int,
        stop: list[str],
    ) -> tuple[str, int]:
        """Greedily generate a deterministic continuation."""

        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync, user, prefix, max_tokens, stop
            )

    async def aclose(self) -> None:
        """Release no resources because the model is owned by the caller."""
