from dataclasses import dataclass
from collections import deque
from typing import Protocol
from engine.config import SchedulerConfig
from engine.processor import Request
from uuid import UUID
import logging

logger = logging.getLogger("inference.scheduler")


class SchedulingPolicy(Protocol):
    def select(
        self,
        pending: deque[Request],
        running: dict[UUID, Request],
        batch_size: int,
    ) -> list[Request]:
        ...

class RoundRobinPolicy:
    def select(self, pending, running, batch_size):
        batch = []

        while pending and len(batch) < batch_size:
            batch.append(pending.popleft())

        return batch


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig
    ):
        self.batch_size = config.batch_size
        self.policy = RoundRobinPolicy()
        
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

    def add_request(self, request: Request):
        logger.debug(
            "scheduler_enqueue request_id=%s",
            request.request_id,
        )
        self.pending.append(request)

    def finish_step(self, request: Request, *, finished: bool):
        self.running.pop(request.request_id, None)
        if not finished:
            self.pending.append(request)