from engine.processor.request import Request
import torch
class Runner:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(self.device)

    def run(self, batch: list[list[int]]) -> list[int]:
        input_ids = torch.tensor(
            batch, 
            dtype=torch.int, 
            device=self.device
        )
        logits = self.model(input_ids)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        return next_tokens.tolist()

