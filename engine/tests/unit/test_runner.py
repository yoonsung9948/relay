import torch
import pytest

from engine.sampler import GreedySampler
from engine.tests.fakes.fake_model import FakeModel
from engine.runner import Runner



@pytest.mark.parametrize(
    ("batch", "logits", "expected"),
    [
        (
            [[1, 2, 3]],
            [
                [0.1, 0.2, 5.0, 0.3],
            ],
            [2],
        ),
        (
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [0.1, 0.2, 5.0, 0.3],
                [0.5, 0.1, 3.0, 0.2],
            ],
            [2, 2],
        ),
    ],
)
def test_runner_runs_and_samples(
    batch,
    logits,
    expected,
    device,
    sampler,
):
    logits = torch.tensor(logits, device=device)

    model = FakeModel(logits)
    runner = Runner(model, device, sampler)

    actual = runner.run(batch)

    assert model.call_count == 1
    assert model.last_input is not None

    torch.testing.assert_close(
        model.last_input,
        torch.tensor(batch, device=device),
    )

    assert actual == expected

    
def test_runner_output_match_shape():
    pass