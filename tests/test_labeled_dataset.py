
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.active_learning.label_store import LabelStore
from src.active_learning.labeled_dataset import LabeledDataset


def main():

    # Create a small dummy dataset
    images = torch.randn(10, 3, 32, 32)

    hidden_labels = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    ])

    base_dataset = TensorDataset(
        images,
        hidden_labels,
    )

    # Initialize Label Store
    store = LabelStore()

    # Only three samples are annotated
    store.add({
        1: 7,
        4: 2,
        8: 5,
    })

    labeled_indices = [1, 4, 8]

    # Create wrapped dataset
    labeled_dataset = LabeledDataset(
        base_dataset=base_dataset,
        label_store=store,
        indices=labeled_indices,
    )

    assert len(labeled_dataset) == 3

    # Verify labels come from Label Store
    _, label_1 = labeled_dataset[0]
    _, label_2 = labeled_dataset[1]
    _, label_3 = labeled_dataset[2]

    assert label_1 == 7
    assert label_2 == 2
    assert label_3 == 5

    print("--- Labeled Dataset Test ---")
    print("Dataset size:", len(labeled_dataset))
    print(
        "Retrieved labels:",
        [label_1, label_2, label_3],
    )

    # Verify unannotated samples are rejected
    try:
        LabeledDataset(
            base_dataset=base_dataset,
            label_store=store,
            indices=[1, 3],
        )

    except ValueError:
        print(
            "Unannotated sample protection passed!"
        )

    else:
        raise AssertionError(
            "Unannotated sample was accepted."
        )

    # Verify DataLoader compatibility
    loader = DataLoader(
        labeled_dataset,
        batch_size=2,
        shuffle=False,
    )

    images_batch, labels_batch = next(iter(loader))

    assert images_batch.shape == (2, 3, 32, 32)
    assert labels_batch.tolist() == [7, 2]

    print("DataLoader compatibility passed!")
    print("\nAll Labeled Dataset tests passed!")


if __name__ == "__main__":
    main()