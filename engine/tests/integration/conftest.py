import pytest
import torch

from engine.model import HuggingFaceModel
from engine.config import Qwen3Config
from engine.model.qwen3 import Qwen3Model


@pytest.fixture()
def model(qwen_config):
    model = Qwen3Model(
        qwen_config,
    )
    return model



@pytest.fixture(scope="session")
def ref_model(model_repo):
    model = HuggingFaceModel(
        model_repo,
        torch.float32,
    )

    return model