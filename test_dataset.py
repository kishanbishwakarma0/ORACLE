
from src.data.dataset import CIFAR10DataModule


def main():
    data_module = CIFAR10DataModule(
        data_dir="data/raw",
        batch_size=64,
        initial_labeled_size=1000,
    )

    data_module.prepare_data()
    data_module.setup()
    data_module.validate()

    (
        labeled_loader,
        unlabeled_loader,
        validation_loader,
        test_loader,
    ) = data_module.get_loaders()

    images, labels = next(iter(labeled_loader))

    print("\n--- DataLoader Test ---")

    images, labels = next(iter(labeled_loader))

    print(f"Labeled batch image shape: {images.shape}")
    print(f"Labeled batch label shape: {labels.shape}")

    print(f"Unlabeled batches: {len(unlabeled_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\nORACLE data pipeline is working!")


if __name__ == "__main__":
    main()