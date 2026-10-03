import pytest
import torch



@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model_repo():
    return "Qwen/Qwen3-0.6B"


