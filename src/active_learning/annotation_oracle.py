
"""
ORACLE - Annotation Oracle

Simulates the process of revealing labels for
selected samples in an active learning experiment.

The sampler receives only images and predictions.
Ground-truth labels are revealed only after selection.
"""


class AnnotationOracle:
    def __init__(self, dataset):
        """
        Args:
            dataset: Training dataset containing (image, label).
        """

        self.dataset = dataset
        self.annotated_indices = set()
        self.annotation_history = []

    def annotate(self, selected_indices):
        """
        Reveal labels for selected samples.

        Returns:
            Dictionary mapping dataset indices to labels.
        """

        annotations = {}

        for index in selected_indices:

            if index in self.annotated_indices:
                raise ValueError(
                    f"Sample {index} has already been annotated."
                )

            _, label = self.dataset[index]

            if hasattr(label, "item"):
                label = label.item()

            annotations[index] = int(label)

            self.annotated_indices.add(index)

            self.annotation_history.append({
                "index": index,
                "label": int(label),
            })

        return annotations

    
    def register_existing(self, indices):
        """
        Register samples that were labeled initially.
        Their labels are already available before the experiment.
        """

        for index in indices:
            if index in self.annotated_indices:
                raise ValueError(
                    f"Sample {index} is already registered."
                )

            self.annotated_indices.add(index)
            
    def get_annotation_count(self):
        """Return total number of annotated samples."""

        return len(self.annotated_indices)

    def get_history(self):
        """Return annotation history."""

        return self.annotation_history.copy()