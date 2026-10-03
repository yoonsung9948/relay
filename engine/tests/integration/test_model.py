import torch
import pytest
from engine.model import load_custom_qwen3
from engine.model.qwen3 import qwen3_06b_config


@pytest.mark.parametrize(
    "batch",
    [
        [[1, 2, 3]],
        [[1, 2, 3], [4, 5, 6]],
    ],
)
def test_model_output_against_ref_model(
    ref_model,
    model_repo,
    batch,
    device
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config = qwen3_06b_config()
    test_model = load_custom_qwen3(
        model_repo,
        config,
        torch.float32,
    )

    ref_model.eval()
    test_model.eval()

    input_ids = torch.tensor(
        batch,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        want = ref_model(input_ids)
        got = test_model(input_ids)

    torch.testing.assert_close(
        got,
        want,
        rtol=1e-3,
        atol=1e-3,
    )