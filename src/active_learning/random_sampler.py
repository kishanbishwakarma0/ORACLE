
"""
ORACLE - Random Sampling

Selects samples randomly from the unlabeled pool.

This serves as the baseline strategy for comparing
active learning algorithms.
"""

import random


class RandomSampler:
    """Randomly selects samples from an unlabeled pool."""

    def __init__(self, pool_indices, seed=42):
        """
        Args:
            pool_indices: Indices of currently unlabeled samples.
            seed: Random seed for reproducibility.
        """

        self.pool_indices = list(pool_indices)
        self.rng = random.Random(seed)

    def query(self, n_samples):
        """
        Randomly select samples from the unlabeled pool.

        Returns:
            List of selected sample indices.
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if n_samples > len(self.pool_indices):
            raise ValueError(
                "Requested samples exceed the available pool."
            )

        selected_indices = self.rng.sample(
            self.pool_indices,
            n_samples,
        )

        selected_set = set(selected_indices)

        self.pool_indices = [
            idx
            for idx in self.pool_indices
            if idx not in selected_set
        ]

        return selected_indices

    def get_pool_size(self):
        """Return the number of remaining unlabeled samples."""

        return len(self.pool_indices)