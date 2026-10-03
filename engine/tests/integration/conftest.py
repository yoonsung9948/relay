import pytest
import torch

from engine.model import HuggingFaceModel, load_custom_qwen3
from engine.model.qwen3 import Qwen3Model, RMSNorm


@pytest.fixture(scope="session")
def qwen_config():
    from engine.model.qwen3 import qwen3_06b_config
    return qwen3_06b_config()

@pytest.fixture(scope="session")
def model(qwen_config):
    model = Qwen3Model(
        qwen_config,
    )
    return model

@pytest.fixture(scope="session")
def qwen_model(model_repo, qwen_config, dtype):
    test_model = load_custom_qwen3(
        model_repo,
        qwen_config,
        dtype,
    )
    return test_model

@pytest.fixture(scope="session")
def ref_model(model_repo, dtype):
    model = HuggingFaceModel(
        model_repo,
        dtype,
    )

    return model
