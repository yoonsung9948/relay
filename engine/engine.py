# inference/engine.py
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

        while request.generated_tokens < request.max_tokens:
            next_token = self.runner.run(request.tokens)

            request.tokens.append(next_token)
            request.generated_tokens += 1

            if next_token in self.eos_token_ids:
                break

        return GenerateResult(
            request_id=request.request_id,
            token_ids=request.tokens,
            generated_tokens=request.tokens[prompt_length:],
        )
    
    async def shutdown(self):
        ...