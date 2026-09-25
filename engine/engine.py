# inference/engine.py
from engine.scheduler import Scheduler
from engine.runner import Runner
from engine.processor import Request

class Engine:
    def __init__(
        self,
        scheduler: Scheduler,
        runner: Runner,
    ):
        self.scheduler = scheduler
        self.runner = runner
    async def generate(self, request: Request):
        ...

    async def shutdown(self):
        ...