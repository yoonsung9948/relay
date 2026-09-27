import logging
import time

import torch

logger = logging.getLogger("inference.runner")


class Runner:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self._first_forward = True

    def run(self, batch: list[list[int]]) -> list[int]:
        first_forward = self._first_forward
        started = time.perf_counter()
        if first_forward:
            logger.info(
                "first_forward_begin model=%s device=%s batch_size=%d sequence_lengths=%s",
                type(self.model).__name__, self.device, len(batch),
                [len(tokens) for tokens in batch],
            )
        input_ids = torch.tensor(batch, dtype=torch.int, device=self.device)
        logits = self.model(input_ids)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        result = next_tokens.tolist()
        if first_forward:
            logger.info(
                "first_forward_complete logits_shape=%s dtype=%s elapsed_s=%.3f",
                tuple(logits.shape), logits.dtype, time.perf_counter() - started,
            )
            self._first_forward = False
        return result
