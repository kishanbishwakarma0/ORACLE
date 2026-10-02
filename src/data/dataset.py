
"""
ORACLE - Dataset Management Module

Handles:
- CIFAR-10 dataset downloading
- Labeled and unlabeled data separation
- Validation and test dataset preparation
- Reproducible dataset splitting
- Dataset validation
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class CIFAR10DataModule:
    """Manages CIFAR-10 data for ORACLE experiments."""

    def __init__(
        self,
        data_dir="data/raw",
        batch_size=64,
        initial_labeled_size=1000,
        validation_size=5000,
        num_workers=0,
        seed=42,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.initial_labeled_size = initial_labeled_size
        self.validation_size = validation_size
        self.num_workers = num_workers
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        self.train_dataset = None
        self.test_dataset = None
        self.labeled_dataset = None
        self.unlabeled_dataset = None
        self.validation_dataset = None

    def prepare_data(self):
        """Download CIFAR-10 training and test datasets."""

        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )

        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform,
        )

        print("CIFAR-10 datasets downloaded successfully.")

    def setup(self):
        """Create reproducible labeled, unlabeled and validation subsets."""

        if self.train_dataset is None:
            self.prepare_data()

        total_samples = len(self.train_dataset)

        if self.validation_size <= 0:
            raise ValueError("validation_size must be positive.")

        if self.initial_labeled_size <= 0:
            raise ValueError("initial_labeled_size must be positive.")

        if self.initial_labeled_size + self.validation_size >= total_samples:
            raise ValueError(
                "The combined labeled and validation sizes "
                "must be smaller than the training dataset."
            )

        generator = torch.Generator().manual_seed(self.seed)

        indices = torch.randperm(
            total_samples,
            generator=generator,
        ).tolist()

        validation_indices = indices[:self.validation_size]

        labeled_start = self.validation_size
        labeled_end = labeled_start + self.initial_labeled_size

        labeled_indices = indices[labeled_start:labeled_end]
        unlabeled_indices = indices[labeled_end:]

        self.validation_dataset = Subset(
            self.train_dataset,
            validation_indices,
        )

        self.labeled_dataset = Subset(
            self.train_dataset,
            labeled_indices,
        )

        self.unlabeled_dataset = Subset(
            self.train_dataset,
            unlabeled_indices,
        )

        print("Dataset split completed.")

    def get_loaders(self):
        """Return DataLoaders for training and evaluation."""

        if self.labeled_dataset is None:
            self.setup()

        labeled_loader = DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return (
            labeled_loader,
            unlabeled_loader,
            validation_loader,
            test_loader,
        )

    def validate(self):
        """Print dataset statistics and validate sample shapes."""

        if self.labeled_dataset is None:
            self.setup()

        image, label = self.labeled_dataset[0]

        print("\n--- ORACLE Dataset Report ---")
        print(f"Total training samples: {len(self.train_dataset)}")
        print(f"Labeled samples: {len(self.labeled_dataset)}")
        print(f"Unlabeled samples: {len(self.unlabeled_dataset)}")
        print(f"Validation samples: {len(self.validation_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Image shape: {tuple(image.shape)}")
        print(f"Sample label: {label}")
        print(f"Number of classes: {len(self.train_dataset.classes)}")
        print(f"Classes: {self.train_dataset.classes}")

        assert image.shape == (3, 32, 32)
        assert 0 <= label < 10

        assert (
            len(self.labeled_dataset)
            + len(self.unlabeled_dataset)
            + len(self.validation_dataset)
            == len(self.train_dataset)
        )

        print("\nDataset validation passed!")