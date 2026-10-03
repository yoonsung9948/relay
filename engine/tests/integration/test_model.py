import torch
import pytest
from engine.model import load_custom_qwen3
from engine.model.qwen3 import qwen3_06b_config, RMSNorm

@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch",
    [
        [[1, 2, 3]],
        [[1, 2, 3], [4, 5, 6]],
    ],
)
def test_model_logits_against_ref_model(
    ref_model,
    qwen_model,
    qwen_config,
    batch,
    device
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    ref_model.eval()
    qwen_model.eval()

    input_ids = torch.tensor(
        batch,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        want = ref_model(input_ids)
        got = qwen_model(input_ids)

    assert got.shape == (len(batch), len(batch[0]), qwen_config.vocab_size)
    torch.testing.assert_close(
        got,
        want,
        rtol=1e-3,
        atol=1e-3,
    )

@pytest.mark.gpu
@pytest.mark.parametrize(
    "input",
    [
        [[1.0, 2.0, 3.0]],
        [[4.0, 5.0, 6.0]],
    ],
)
def test_model_rms_norm(device, input, dtype):
    input_tensor = torch.tensor(
        input,
        dtype=dtype,
        device=device,
    )
    rms_norm = RMSNorm(input_tensor.shape[-1]).to(device=device, dtype=dtype)
    with torch.no_grad():
        rms_norm.weight.copy_(torch.ones_like(rms_norm.weight))

    output = rms_norm(input_tensor)
    output = output.to(dtype=dtype)
    assert output.shape == input_tensor.shape
    assert output.dtype == dtype
    
    variance = input_tensor.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + rms_norm.eps)

    expected = normalized * rms_norm.weight

    torch.testing.assert_close(output, expected)
    