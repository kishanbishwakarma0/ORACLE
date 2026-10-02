
"""
ORACLE - Label Store

Stores labels only for samples that have been
explicitly annotated.

This component separates the labeled dataset
from the original dataset's hidden labels.
"""


class LabelStore:

    def __init__(self):
        self.labels = {}

    def add(self, annotations):
        """
        Add newly annotated labels.

        Args:
            annotations: Dictionary {index: label}
        """

        for index, label in annotations.items():

            if index in self.labels:
                raise ValueError(
                    f"Label for sample {index} already exists."
                )

            self.labels[index] = int(label)

    def get_label(self, index):
        """Retrieve an annotated label."""

        if index not in self.labels:
            raise KeyError(
                f"Sample {index} has not been annotated."
            )

        return self.labels[index]

    def is_annotated(self, index):
        """Check whether a sample has an available label."""

        return index in self.labels

    def get_annotated_indices(self):
        """Return all annotated sample indices."""

        return list(self.labels.keys())

    def get_annotation_count(self):
        """Return the number of stored labels."""

        return len(self.labels)

    def get_all_labels(self):
        """Return a copy of stored annotations."""

        return self.labels.copy()