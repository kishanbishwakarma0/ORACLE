
"""
ORACLE - Uncertainty Sampling

Least Confidence strategy selects samples for which
the model has the lowest maximum class probability.

This implementation works with UnlabeledImageDataset,
which exposes images without ground-truth labels.
"""

import torch
from torch.utils.data import DataLoader


class UncertaintySampler:
    """
    Selects the least confident samples from an
    unlabeled pool.
    """

    def __init__(self, pool_indices):

        self.pool_indices = list(pool_indices)

        if len(self.pool_indices) != len(set(self.pool_indices)):
            raise ValueError(
                "Duplicate indices found in unlabeled pool."
            )

    def query(
        self,
        model,
        dataset,
        n_samples,
        device,
        batch_size=64,
    ):
        """
        Select the least confident samples.

        Args:
            model:
                Trained classification model.

            dataset:
                UnlabeledImageDataset containing only images.
                Its order must match self.pool_indices.

            n_samples:
                Number of samples to select.

            device:
                CPU or CUDA device.

            batch_size:
                Inference batch size.

        Returns:
            Selected original dataset indices.
        """

        if n_samples <= 0:
            raise ValueError(
                "n_samples must be positive."
            )

        if n_samples > len(self.pool_indices):
            raise ValueError(
                "Requested samples exceed the available pool."
            )

        if len(dataset) != len(self.pool_indices):
            raise ValueError(
                "Dataset size must match the sampler pool size."
            )

        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_uncertainties = []

        with torch.no_grad():

            for images in loader:

                images = images.to(device)

                outputs = model(images)

                probabilities = torch.softmax(
                    outputs,
                    dim=1,
                )

                max_probabilities = probabilities.max(
                    dim=1
                ).values

                # Least confidence:
                # Uncertainty = 1 - maximum class probability
                uncertainties = (
                    1.0 - max_probabilities
                )

                all_uncertainties.extend(
                    uncertainties.cpu().tolist()
                )

        if len(all_uncertainties) != len(self.pool_indices):
            raise RuntimeError(
                "Uncertainty score count does not match pool size."
            )

        uncertainty_tensor = torch.tensor(
            all_uncertainties,
            dtype=torch.float32,
        )

        # Select samples with highest uncertainty.
        selected_positions = torch.topk(
            uncertainty_tensor,
            k=n_samples,
        ).indices.tolist()

        # Map selected positions to original dataset indices.
        selected_indices = [
            self.pool_indices[position]
            for position in selected_positions
        ]

        selected_set = set(selected_indices)

        # Remove selected samples from the pool.
        self.pool_indices = [
            index
            for index in self.pool_indices
            if index not in selected_set
        ]

        return selected_indices

    def get_pool_size(self):
        """Return the number of remaining unlabeled samples."""

        return len(self.pool_indices)