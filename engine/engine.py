# inference/engine.py
import asyncio

from engine.scheduler import Scheduler
from engine.runner import Runner
from engine.processor import Request, GenerateResult

class Engine:
    def __init__(
        self,
        scheduler: Scheduler,
        runner: Runner,
        eos: int | list[int] | None
    ):
        self.scheduler = scheduler
        self.runner = runner
        self.eos_token_ids = frozenset(
            [] if eos is None else [eos] if isinstance(eos, int) else eos
        )


    async def generate(self, request: Request) -> GenerateResult:
        prompt_length = len(request.tokens)
        self.scheduler.add_request(request)
        await request.done_event.wait()

        return GenerateResult(
            request_id=request.request_id,
            token_ids=request.tokens,
            generated_tokens=len(request.tokens) - prompt_length,
        )


    async def run_loop(self):
        while True:
            batch = self.scheduler.schedule()

            if not batch:
                await asyncio.sleep(0.01)
                continue

            results = self.runner.run([request.tokens for request in batch])
            for request, result in zip(batch, results):
                request.tokens.append(result)
                request.generated_tokens += 1
                self.scheduler.running.pop(request.request_id, None)
                finished = (
                    result in self.eos_token_ids 
                    or request.generated_tokens >= request.max_tokens
                )

                self.scheduler.finish_step(request, finished=finished)

                if finished:
                    request.done_event.set()
            await asyncio.sleep(0)

    async def shutdown(self):
        ...