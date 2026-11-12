import logging
import time
from collections.abc import Sequence

import torch

from engine.config.config import KVPoolConfig
from engine.kv_cache import KVCache
from engine.kv_pool import KVPool, PoolBatch
from engine.processor.request import Request
from engine.sampler import Sampler
from engine.sampling import SamplingOptions

logger = logging.getLogger("inference.runner")


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        sampler: Sampler,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self._first_forward = True
        self.sampler = sampler
        self.pool = None
        self.cache_config = KVPoolConfig()
        self.reservations = {}

    def configure_cache(self, config):
        self.cache_config = config
        if config.slots:
            if not self.supports_kv_cache:
                raise ValueError("KV pool requires the custom Qwen3 backend")
            attention = self.model.model.layers[0].self_attn
            self.pool = KVPool(
                self.model.num_hidden_layers,
                config.slots,
                attention.num_kv_heads,
                config.max_seq_len,
                attention.head_dim,
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
                max_bytes=config.max_bytes,
            )

    def validate_request(self, request):
        if len(request.tokens) + request.max_tokens > self.cache_config.max_seq_len:
            raise ValueError("requested context exceeds max_seq_len")
        if request.kv_cache_mode == "pool" and self.pool is None:
            raise ValueError("KV pool is not enabled in server configuration")
        if request.kv_cache_mode == "dynamic":
            needed = self._dynamic_reservation(request)
            if needed > self.cache_config.dynamic_max_bytes:
                raise ValueError("request exceeds dynamic KV budget")

    def _dynamic_reservation(self, request):
        attention = self.model.model.layers[0].self_attn
        per_token = (
            2
            * self.model.num_hidden_layers
            * attention.num_kv_heads
            * attention.head_dim
            * next(self.model.parameters()).element_size()
        )
        # Include headroom for old and new tensors during torch.cat.
        return 2 * per_token * (len(request.tokens) + request.max_tokens)

    def admit(self, request):
        if request.kv_cache_mode == "pool":
            handle = self.pool.allocate(request.request_id)
            if handle is None:
                return False
            request.kv_handle = handle
        elif request.kv_cache_mode == "dynamic":
            needed = self._dynamic_reservation(request)
            if sum(self.reservations.values()) + needed > self.cache_config.dynamic_max_bytes:
                return False
            self.reservations[request.request_id] = needed
        return True

    def release(self, request):
        if request.kv_handle is not None:
            self.pool.release(request.kv_handle)
            request.kv_handle = None
        self.reservations.pop(request.request_id, None)

    @property
    def supports_kv_cache(self) -> bool:
        return getattr(self.model, "supports_kv_cache", False) is True

    def run(self, batch: list[list[int]]) -> list[int]:
        return self._run(batch)

    def run_requests(self, batch: list[Request]) -> list[int]:
        options = [request.sampling for request in batch]
        if all(request.kv_cache_mode == "pool" for request in batch):
            context = self.pool.batch([request.kv_handle for request in batch])
            inputs = [req.tokens[previous:] for req, previous in zip(batch, context.previous)]
            if len(batch) > 1 and any(len(tokens) != 1 for tokens in inputs):
                raise ValueError("only single-token decode can be batched")
            result = self._run(inputs, cache=context, options=options)
            context.commit()
            for req, previous, tokens in zip(batch, context.previous, inputs):
                req.cached_tokens = previous + len(tokens)
            return result
        if not any(request.use_kv_cache for request in batch):
            return self._run([request.tokens for request in batch], options=options)
        if len(batch) != 1:
            raise ValueError("cached execution currently requires batch_size=1")
        if not self.supports_kv_cache:
            raise ValueError("KV cache is supported only by the custom Qwen3 backend")
        request = batch[0]
        if request.kv_cache is None:
            request.kv_cache = KVCache(self.model.num_hidden_layers)
        past_length = request.kv_cache.get_seq_length()
        input_tokens = request.tokens[past_length:]
        if not input_tokens:
            raise ValueError("cached inference requires at least one uncached input token")
        return self._run([input_tokens], cache=request.kv_cache, options=options)

    @torch.inference_mode()
    def _run(
        self,
        batch: list[list[int]],
        cache: KVCache | PoolBatch | None = None,
        options: Sequence[SamplingOptions] | None = None,
    ) -> list[int]:
        first_forward = self._first_forward
        started = time.perf_counter()
        if first_forward:
            logger.info(
                "first_forward_begin model=%s device=%s batch_size=%d sequence_lengths=%s",
                type(self.model).__name__,
                self.device,
                len(batch),
                [len(tokens) for tokens in batch],
            )
        input_ids = torch.tensor(batch, dtype=torch.long, device=self.device)
        logits = self.model(input_ids) if cache is None else self.model(input_ids, cache=cache)
        # Each request needs the prediction after its final input token.
        next_tokens = self.sampler.sample(logits[:, -1, :], options)
        result = next_tokens.tolist()
        if first_forward:
            logger.info(
                "first_forward_complete logits_shape=%s dtype=%s elapsed_s=%.3f",
                tuple(logits.shape),
                logits.dtype,
                time.perf_counter() - started,
            )
            self._first_forward = False
        return result
