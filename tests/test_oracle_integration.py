
import torch
from torch.utils.data import TensorDataset

from src.active_learning.annotation_oracle import AnnotationOracle
from src.active_learning.random_sampler import RandomSampler


def main():

    images = torch.randn(20, 3, 32, 32)
    labels = torch.randint(0, 10, (20,))

    dataset = TensorDataset(images, labels)

    initial_indices = list(range(5))
    unlabeled_indices = list(range(5, 20))

    oracle = AnnotationOracle(dataset)

    oracle.register_existing(initial_indices)

    sampler = RandomSampler(
        pool_indices=unlabeled_indices,
        seed=42,
    )

    print("Initial annotations:", oracle.get_annotation_count())

    selected = sampler.query(5)

    annotations = oracle.annotate(selected)

    print("Newly annotated:", len(annotations))
    print("Total annotations:", oracle.get_annotation_count())

    assert oracle.get_annotation_count() == 10

    assert len(set(initial_indices) & set(selected)) == 0

    print("\nOracle integration test passed!")


if __name__ == "__main__":
    main()