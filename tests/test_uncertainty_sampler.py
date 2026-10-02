
"""
Tests for UncertaintySampler.

Verifies:
- Correct uncertainty ranking
- Original index mapping
- Pool size updates
- Image-only dataset compatibility
- Invalid query protection
"""

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset

from src.active_learning.uncertainty_sampler import (
    UncertaintySampler,
)

from src.data.unlabeled_dataset import (
    UnlabeledImageDataset,
)


class DummyModel(nn.Module):
    """
    Dummy model that generates predictable
    confidence scores using image pixel values.
    """

    def forward(self, x):

        values = x[:, 0, 0, 0]

        logits = torch.stack(
            [
                values,
                torch.zeros_like(values),
            ],
            dim=1,
        )

        return logits


def main():

    print("--- Uncertainty Sampler Test ---")

    # Create 5 samples with predictable model outputs.
    images = torch.tensor(
        [
            [[[5.0]]],
            [[[0.1]]],
            [[[3.0]]],
            [[[0.2]]],
            [[[4.0]]],
        ]
    )

    labels = torch.tensor(
        [0, 1, 0, 1, 0]
    )

    base_dataset = TensorDataset(
        images,
        labels,
    )

    # Original dataset indices.
    pool_indices = [10, 20, 30, 40, 50]

    # Sampler sees only images.
    unlabeled_dataset = UnlabeledImageDataset(
        base_dataset=base_dataset,
        indices=[0, 1, 2, 3, 4],
    )

    sampler = UncertaintySampler(
        pool_indices=pool_indices,
    )

    model = DummyModel()

    selected = sampler.query(
        model=model,
        dataset=unlabeled_dataset,
        n_samples=2,
        device=torch.device("cpu"),
        batch_size=2,
    )

    # Samples 20 and 40 have the lowest confidence.
    assert set(selected) == {20, 40}

    # Two samples should be removed.
    assert sampler.get_pool_size() == 3

    # Selected samples must not remain in the pool.
    assert not (
        set(selected)
        & set(sampler.pool_indices)
    )

    print("Uncertainty ranking passed!")
    print("Original index mapping passed!")
    print("Pool update passed!")
    print("Image-only dataset compatibility passed!")

    # Invalid query test.
    try:

        sampler.query(
            model=model,
            dataset=unlabeled_dataset,
            n_samples=0,
            device=torch.device("cpu"),
        )

        raise AssertionError(
            "Invalid query protection failed."
        )

    except ValueError:
        pass

    print("Invalid query protection passed!")

    print(
        "\nAll uncertainty sampler tests passed!"
    )


if __name__ == "__main__":
    main()