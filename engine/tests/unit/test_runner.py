import pytest
import torch

from engine.runner import Runner
from engine.sampler import GreedySampler
from engine.tests.fakes.fake_model import FakeModel


@pytest.mark.parametrize(
    ("batch", "logits", "expected"),
    [
        (
            [[1, 2, 3]],
            [[[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.1, 0.2, 5.0, 0.3]]],
            [2],
        ),
        (
            [[1, 2, 3], [4, 5, 6]],
            [
                [[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.1, 0.2, 5.0, 0.3]],
                [[0.0, 0.0, 5.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.5, 0.1, 0.2, 3.0]],
            ],
            [2, 3],
        ),
    ],
)
def test_runner_samples_only_the_last_sequence_position(batch, logits, expected, device):
    model = FakeModel(torch.tensor(logits, device=device))
    runner = Runner(model, device, GreedySampler())

    actual = runner.run(batch)

    assert model.call_count == 1
    torch.testing.assert_close(
        model.last_input,
        torch.tensor(batch, device=device),
    )
    assert actual == expected
    assert all(type(token) is int for token in actual)
