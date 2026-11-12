from collections import deque
from typing import Protocol

from engine.config import SchedulerConfig
from engine.execution import CapacityError
from engine.processor import Request


class SchedulingPolicy(Protocol):
    def select(self, pending, running, batch_size) -> list[Request]: ...


class RoundRobinPolicy:
    def select(self, pending, running, batch_size):
        return [pending.popleft() for _ in range(min(len(pending), batch_size))]


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.policy = RoundRobinPolicy()
        self.pending = deque()
        self.ready = deque()
        self.running = {}
        self.active = {}
        self.admit = lambda request: True
        self._phase = 0
        self.rejected_requests = 0

    def add_request(self, request):
        if len(self.pending) >= self.config.max_pending_requests:
            self.rejected_requests += 1
            raise CapacityError("generation waiting queue is full")
        self.pending.append(request)

    @staticmethod
    def phase(request):
        if request.generated_tokens == 0:
            return "prefill"
        if request.kv_cache_mode == "pool" and request.decode_batching:
            return "decode"
        return "baseline"

    def schedule(self):
        for _ in range(len(self.pending)):
            request = self.pending.popleft()
            if request.cancelled:
                continue
            if len(self.active) < self.config.max_active_requests and self.admit(request):
                self.active[request.request_id] = request
                self.ready.append(request)
            else:
                self.pending.append(request)
        phases = ("decode", "prefill", "baseline")
        for _ in phases:
            phase = phases[self._phase]
            self._phase = (self._phase + 1) % len(phases)
            eligible = deque(
                req for req in self.ready if self.phase(req) == phase and not req.cancelled
            )
            if not eligible:
                continue
            limit = self.config.max_decode_batch_size if phase == "decode" else 1
            batch = self.policy.select(eligible, self.running, limit)
            selected = {req.request_id for req in batch}
            self.ready = deque(req for req in self.ready if req.request_id not in selected)
            self.running.update((req.request_id, req) for req in batch)
            return batch
        return []

    def cancel_request(self, request):
        request.cancelled = True
        self.finish_step(request, finished=True)
        self.pending = deque(req for req in self.pending if req.request_id != request.request_id)

    def finish_step(self, request, *, finished):
        self.running.pop(request.request_id, None)
        self.ready = deque(req for req in self.ready if req.request_id != request.request_id)
        if finished or request.cancelled:
            self.active.pop(request.request_id, None)
        else:
            self.ready.append(request)

    def clear(self):
        self.pending.clear()
        self.ready.clear()
        self.running.clear()
        self.active.clear()
