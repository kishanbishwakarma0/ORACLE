
import torch
from torch.utils.data import TensorDataset

from src.active_learning.annotation_oracle import AnnotationOracle


def main():

    images = torch.randn(5, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4])

    dataset = TensorDataset(images, labels)

    oracle = AnnotationOracle(dataset)

    selected = [1, 3]

    annotations = oracle.annotate(selected)

    assert annotations == {
        1: 1,
        3: 3,
    }

    assert oracle.get_annotation_count() == 2

    print("--- Annotation Oracle Test ---")
    print("Annotations:", annotations)
    print("Annotation count:", oracle.get_annotation_count())
    print("Annotation history:", oracle.get_history())

    # Verify duplicate annotation protection
    try:
        oracle.annotate([1])
    except ValueError:
        print("Duplicate annotation protection passed!")
    else:
        raise AssertionError(
            "Duplicate annotation was not blocked."
        )

    print("\nAll Annotation Oracle tests passed!")


if __name__ == "__main__":
    main()