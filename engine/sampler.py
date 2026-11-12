import math
from collections.abc import Sequence
from typing import Protocol

import torch

from engine.sampling import SamplingOptions


class Sampler(Protocol):
    def sample(
        self, logits: torch.Tensor, options: Sequence[SamplingOptions] | None = None
    ) -> torch.Tensor: ...


class GreedySampler:
    def sample(
        self, logits: torch.Tensor, options: Sequence[SamplingOptions] | None = None
    ) -> torch.Tensor:
        if options is not None and any(option.temperature != 0 for option in options):
            raise ValueError("GreedySampler requires zero temperature")
        return logits.argmax(dim=-1)


class TemperatureSampler:
    """Sample each row independently; zero-temperature rows consume no randomness."""

    def sample(
        self,
        logits: torch.Tensor,
        options: Sequence[SamplingOptions] | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("sampling requires [batch, vocabulary] logits")
        if options is not None and len(options) != logits.shape[0]:
            raise ValueError("sampling options must match the batch size")
        result = logits.argmax(dim=-1)
        if options is None:
            return result
        rows = [i for i, option in enumerate(options) if option.temperature > 0]
        if not rows:
            return result

        indices = torch.tensor(rows, dtype=torch.long, device=logits.device)
        temperatures = [options[i].temperature for i in rows]
        scores = logits.index_select(0, indices).float()
        # Center before scaling to avoid overflowing large positive logits at low T.
        scores = scores - scores.amax(dim=-1, keepdim=True)
        limits = torch.finfo(torch.float32)
        if all(limits.tiny <= temperature <= limits.max for temperature in temperatures):
            scale = torch.tensor(temperatures, dtype=torch.float32, device=logits.device)
            scores = scores / scale[:, None]
        else:
            # Preserve finite float64 HTTP values that under/overflow a float32 divisor.
            # Zero gaps remain zero; very large gaps saturate to -inf before softmax.
            log_scale = torch.tensor(
                [math.log(temperature) for temperature in temperatures],
                dtype=torch.float32,
                device=logits.device,
            )
            scores = -torch.exp(torch.log(-scores) - log_scale[:, None])
        probabilities = torch.softmax(scores, dim=-1)
        try:
            sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        except RuntimeError as exc:
            raise RuntimeError(
                "temperature sampling failed; check logits and probabilities"
            ) from exc
        result.index_copy_(0, indices, sampled)
        return result
