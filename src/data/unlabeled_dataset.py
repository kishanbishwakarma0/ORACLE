
"""
ORACLE - Unlabeled Image Dataset

Exposes images without exposing ground-truth labels.

This wrapper is intended for active learning samplers
that need to inspect unlabeled samples without accessing
their labels.
"""

from torch.utils.data import Dataset


class UnlabeledImageDataset(Dataset):
    """
    Dataset wrapper that returns images only.

    Args:
        base_dataset: Original dataset.
        indices: Original indices of unlabeled samples.

    Returns:
        Image only (no label).
    """

    def __init__(self, base_dataset, indices):

        self.base_dataset = base_dataset
        self.indices = list(indices)

        if len(self.indices) != len(set(self.indices)):
            raise ValueError(
                "Duplicate indices found in unlabeled dataset."
            )

        if not self.indices:
            raise ValueError(
                "Unlabeled dataset cannot be empty."
            )

        dataset_size = len(self.base_dataset)

        for index in self.indices:

            if index < 0 or index >= dataset_size:
                raise IndexError(
                    f"Index {index} is outside the dataset."
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        original_index = self.indices[position]

        image, _ = self.base_dataset[original_index]

        return image

    def get_original_index(self, position):
        """Return the original dataset index."""

        return self.indices[position]