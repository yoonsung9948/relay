from dataclasses import dataclass
from collections import deque
from typing import Protocol
from engine.config import SchedulerConfig
from engine.processor import Request
from uuid import UUID




class SchedulingPolicy(Protocol):
    def select(
        self,
        pending: deque[Request],
        running: dict[UUID, Request],
        batch_size: int,
    ) -> list[Request]:
        ...

class FifoPolicy:
    def select(self, pending, running, batch_size):
        batch = []

        while pending and len(batch) < batch_size:
            batch.append(pending.popleft())

        return batch


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        policy: SchedulingPolicy,
    ):
        self.batch_size = config.batch_size
        self.policy = policy

        self.pending: deque[Request] = deque()
        self.running: dict[UUID, Request] = {}

    def schedule(self) -> list[Request]:
        batch = self.policy.select(
            self.pending,
            self.running,
            self.batch_size,
        )

        for req in batch:
            self.running[req.request_id] = req

        return batch