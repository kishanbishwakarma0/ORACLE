
"""
ORACLE - Labeled Dataset Wrapper

Provides images from the original dataset and labels
only from the Label Store.

This prevents the training pipeline from directly
accessing hidden ground-truth labels.
"""

from torch.utils.data import Dataset


class LabeledDataset(Dataset):

    def __init__(self, base_dataset, label_store, indices):
        """
        Args:
            base_dataset: Original image dataset.
            label_store: Store containing annotated labels.
            indices: Dataset indices available for training.
        """

        self.base_dataset = base_dataset
        self.label_store = label_store
        self.indices = list(indices)

        # Validate that all samples are annotated
        for index in self.indices:

            if not self.label_store.is_annotated(index):
                raise ValueError(
                    f"Sample {index} is not annotated."
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        # Map local position to original dataset index
        original_index = self.indices[position]

        # Retrieve image only
        image, _ = self.base_dataset[original_index]

        # Retrieve label from Label Store
        label = self.label_store.get_label(
            original_index
        )

        return image, label