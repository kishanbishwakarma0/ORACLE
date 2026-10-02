
"""
Tests for UnlabeledImageDataset.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.data.unlabeled_dataset import UnlabeledImageDataset


def main():

    print("--- Unlabeled Dataset Test ---")

    # Create a small dummy dataset.
    images = torch.randn(20, 3, 32, 32)
    labels = torch.randint(0, 10, (20,))

    base_dataset = TensorDataset(
        images,
        labels,
    )

    selected_indices = [2, 5, 8, 11, 14]

    unlabeled_dataset = UnlabeledImageDataset(
        base_dataset=base_dataset,
        indices=selected_indices,
    )

    # Check dataset length.
    assert len(unlabeled_dataset) == 5

    # Check that only images are returned.
    sample = unlabeled_dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 32, 32)

    # Verify original index mapping.
    assert unlabeled_dataset.get_original_index(0) == 2

    # Verify DataLoader compatibility.
    loader = DataLoader(
        unlabeled_dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (2, 3, 32, 32)

    # Verify duplicate protection.
    try:

        UnlabeledImageDataset(
            base_dataset=base_dataset,
            indices=[1, 1, 2],
        )

        raise AssertionError(
            "Duplicate protection failed."
        )

    except ValueError:
        pass

    print("Image-only access passed!")
    print("Index mapping passed!")
    print("DataLoader compatibility passed!")
    print("Duplicate protection passed!")
    print("All unlabeled dataset tests passed!")


if __name__ == "__main__":
    main()