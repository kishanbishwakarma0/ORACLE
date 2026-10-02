
import torch
from torch.utils.data import TensorDataset

from src.active_learning.random_sampler import RandomSampler
from src.active_learning.annotation_oracle import AnnotationOracle
from src.active_learning.label_store import LabelStore
from src.active_learning.labeled_dataset import LabeledDataset


def main():

    # Create dummy dataset
    images = torch.randn(20, 3, 32, 32)
    labels = torch.randint(0, 10, (20,))

    dataset = TensorDataset(images, labels)

    # Initial labeled and unlabeled samples
    initial_indices = list(range(5))
    unlabeled_indices = list(range(5, 20))

    # Initialize components
    oracle = AnnotationOracle(dataset)
    label_store = LabelStore()

    sampler = RandomSampler(
        pool_indices=unlabeled_indices,
        seed=42,
    )

    # Register initial annotations
    oracle.register_existing(initial_indices)

    initial_annotations = {
        index: int(labels[index].item())
        for index in initial_indices
    }

    label_store.add(initial_annotations)

    # Verify initial state
    assert oracle.get_annotation_count() == 5
    assert label_store.get_annotation_count() == 5

    # Query new samples
    selected = sampler.query(5)

    # Annotate selected samples
    annotations = oracle.annotate(selected)

    # Update Label Store
    label_store.add(annotations)

    # Update labeled indices
    labeled_indices = initial_indices + list(
        annotations.keys()
    )

    # Verify synchronization
    assert (
        oracle.get_annotation_count()
        == label_store.get_annotation_count()
        == len(labeled_indices)
        == 10
    )

    # Create training dataset
    training_dataset = LabeledDataset(
        base_dataset=dataset,
        label_store=label_store,
        indices=labeled_indices,
    )

    # Verify dataset
    assert len(training_dataset) == 10

    image, label = training_dataset[0]

    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)

    print("--- Active Learning Pipeline Test ---")
    print("Initial annotations: 5")
    print("New annotations:", len(annotations))
    print("Total annotations:", label_store.get_annotation_count())
    print("Training dataset size:", len(training_dataset))
    print("Image shape:", image.shape)
    print("Sample label:", label)

    print("\nAll pipeline integration tests passed!")


if __name__ == "__main__":
    main()