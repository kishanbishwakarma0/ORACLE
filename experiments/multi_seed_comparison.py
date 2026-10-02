
"""
ORACLE - Multi-Seed Active Learning Comparison

Compares Random Sampling and Uncertainty Sampling
on CIFAR-10 using multiple random seeds.

Features:
- Reproducible dataset splitting
- Image-only unlabeled data access
- AnnotationOracle integration
- LabelStore integration
- LabeledDataset integration
- Fresh model training at every annotation budget
- Validation accuracy tracking
- Final test evaluation
- Mean and standard deviation aggregation
- CSV and JSON result exports
- Model checkpoint saving
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CIFAR10DataModule
from src.data.unlabeled_dataset import UnlabeledImageDataset

from src.models.cnn import BaselineCNN

from src.active_learning.random_sampler import RandomSampler
from src.active_learning.uncertainty_sampler import UncertaintySampler
from src.active_learning.annotation_oracle import AnnotationOracle
from src.active_learning.label_store import LabelStore
from src.active_learning.labeled_dataset import LabeledDataset


# ==================================================
# CONFIGURATION
# ==================================================

SEEDS = [42, 123, 2026]

INITIAL_LABELED = 1000
VALIDATION_SIZE = 5000

QUERY_SIZE = 500
MAX_BUDGET = 5000

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

STRATEGIES = [
    "random",
    "uncertainty",
]

DATA_DIR = "data/raw"

EXPERIMENT_DIR = Path("experiments")
CHECKPOINT_DIR = Path("models")

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# REPRODUCIBILITY
# ==================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# MODEL TRAINING
# ==================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS,
):
    """
    Train a model and return its best validation
    accuracy across all epochs.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    for epoch in range(epochs):

        model.train()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for images, labels in progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()
            optimizer.step()

        # ------------------------------------------
        # Validation
        # ------------------------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                total += labels.size(0)

                correct += (
                    predictions == labels
                ).sum().item()

        val_accuracy = (
            100.0 * correct / total
        )

        best_val_accuracy = max(
            best_val_accuracy,
            val_accuracy,
        )

    return best_val_accuracy


# ==================================================
# TEST EVALUATION
# ==================================================

def evaluate_test(
    model,
    test_loader,
):
    """Evaluate the final trained model."""

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    return 100.0 * correct / total


# ==================================================
# DATASET PREPARATION
# ==================================================

def prepare_experiment_data(seed):
    """
    Prepare CIFAR-10 datasets and reproducible splits.
    """

    data_module = CIFAR10DataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        initial_labeled_size=INITIAL_LABELED,
        validation_size=VALIDATION_SIZE,
        num_workers=0,
        seed=seed,
    )

    data_module.prepare_data()
    data_module.setup()

    (
        labeled_loader,
        unlabeled_loader,
        val_loader,
        test_loader,
    ) = data_module.get_loaders()

    base_train_dataset = (
        data_module.train_dataset
    )

    labeled_indices = list(
        data_module.labeled_dataset.indices
    )

    validation_indices = set(
        data_module.validation_dataset.indices
    )

    labeled_set = set(
        labeled_indices
    )

    all_indices = set(
        range(len(base_train_dataset))
    )

    # Validation samples are excluded from the
    # active learning pool.
    unlabeled_indices = sorted(
        all_indices
        - labeled_set
        - validation_indices
    )

    return {
        "base_train_dataset": base_train_dataset,
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


# ==================================================
# SINGLE EXPERIMENT
# ==================================================

def run_experiment(
    strategy_name,
    seed,
):
    """Run one sampling strategy for one seed."""

    print("\n" + "=" * 65)
    print(
        f"STRATEGY: {strategy_name.upper()} | SEED: {seed}"
    )
    print("=" * 65)

    set_seed(seed)

    # ----------------------------------------------
    # Prepare datasets
    # ----------------------------------------------

    data = prepare_experiment_data(seed)

    base_train_dataset = data[
        "base_train_dataset"
    ]

    labeled_indices = data[
        "labeled_indices"
    ]

    unlabeled_indices = data[
        "unlabeled_indices"
    ]

    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    print(
        f"Initial labeled samples: {len(labeled_indices)}"
    )

    print(
        f"Unlabeled pool: {len(unlabeled_indices)}"
    )

    print(
        f"Validation samples: {len(val_loader.dataset)}"
    )

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    # ----------------------------------------------
    # Initialize sampler
    # ----------------------------------------------

    if strategy_name == "random":

        sampler = RandomSampler(
            pool_indices=unlabeled_indices,
            seed=seed,
        )

    elif strategy_name == "uncertainty":

        sampler = UncertaintySampler(
            pool_indices=unlabeled_indices,
        )

    else:

        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    # ----------------------------------------------
    # Initialize annotation components
    # ----------------------------------------------

    oracle = AnnotationOracle(
        base_train_dataset
    )

    label_store = LabelStore()

    # Register initial annotations.
    oracle.register_existing(
        labeled_indices
    )

    # Initial samples are already annotated.
    initial_annotations = {
        index: base_train_dataset[index][1]
        for index in labeled_indices
    }

    label_store.add(
        initial_annotations
    )

    assert (
        oracle.get_annotation_count()
        == label_store.get_annotation_count()
        == len(labeled_indices)
    )

    print(
        "Annotation system initialized successfully."
    )

    # ----------------------------------------------
    # Active learning loop
    # ----------------------------------------------

    results = []

    budget = len(labeled_indices)

    while budget <= MAX_BUDGET:

        print("\n" + "-" * 55)
        print(
            f"Annotation Budget: {budget}"
        )
        print("-" * 55)

        # A fresh model is trained at every budget.
        set_seed(seed + budget)

        model = BaselineCNN().to(DEVICE)

        # Only annotated labels are available to
        # the training pipeline.
        labeled_dataset = LabeledDataset(
            base_dataset=base_train_dataset,
            label_store=label_store,
            indices=labeled_indices,
        )

        train_loader = DataLoader(
            labeled_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        # ------------------------------------------
        # Train and validate
        # ------------------------------------------

        val_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
        )

        print(
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        results.append({
            "strategy": strategy_name,
            "seed": seed,
            "budget": budget,
            "validation_accuracy": val_accuracy,
            "test_accuracy": np.nan,
        })

        # Stop after reaching final budget.
        if budget >= MAX_BUDGET:
            break

        query_size = min(
            QUERY_SIZE,
            MAX_BUDGET - budget,
        )

        # ------------------------------------------
        # Query new samples
        # ------------------------------------------

        if strategy_name == "random":

            newly_selected = sampler.query(
                query_size
            )

        else:

            # Create an image-only view of the
            # current unlabeled pool.
            #
            # The dataset returns images only.
            # Original labels are not exposed.
            unlabeled_dataset = UnlabeledImageDataset(
                base_dataset=base_train_dataset,
                indices=sampler.pool_indices,
            )

            newly_selected = sampler.query(
                model=model,
                dataset=unlabeled_dataset,
                n_samples=query_size,
                batch_size=BATCH_SIZE,
                device=DEVICE,
            )

        # ------------------------------------------
        # Reveal labels through Oracle
        # ------------------------------------------

        annotations = oracle.annotate(
            newly_selected
        )

        # Add only revealed labels to LabelStore.
        label_store.add(
            annotations
        )

        # Update labeled index list.
        labeled_indices.extend(
            annotations.keys()
        )

        # ------------------------------------------
        # Integrity checks
        # ------------------------------------------

        assert (
            oracle.get_annotation_count()
            == label_store.get_annotation_count()
            == len(labeled_indices)
        ), "Annotation count mismatch!"

        assert (
            len(labeled_indices)
            == len(set(labeled_indices))
        ), "Duplicate labeled indices detected!"

        assert (
            len(annotations) == query_size
        ), "Unexpected annotation count!"

        budget = len(labeled_indices)

        print(
            f"New annotations: {len(annotations)}"
        )

        print(
            f"Total labeled samples: {budget}"
        )

        print(
            f"Remaining unlabeled samples: "
            f"{sampler.get_pool_size()}"
        )

    # ==================================================
    # FINAL MODEL TRAINING
    # ==================================================

    print("\nTraining final model...")

    set_seed(seed + MAX_BUDGET)

    final_model = BaselineCNN().to(DEVICE)

    final_dataset = LabeledDataset(
        base_dataset=base_train_dataset,
        label_store=label_store,
        indices=labeled_indices,
    )

    final_loader = DataLoader(
        final_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    train_model(
        model=final_model,
        train_loader=final_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
    )

    # ----------------------------------------------
    # Final test evaluation
    # ----------------------------------------------

    test_accuracy = evaluate_test(
        final_model,
        test_loader,
    )

    print(
        f"Final Test Accuracy: {test_accuracy:.2f}%"
    )

    results[-1]["test_accuracy"] = (
        test_accuracy
    )

    # ----------------------------------------------
    # Save checkpoint
    # ----------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{strategy_name}_seed_{seed}_final.pth"
    )

    torch.save(
        final_model.state_dict(),
        checkpoint_path,
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )

    return results


# ==================================================
# MAIN EXPERIMENT
# ==================================================

def main():

    start_time = time.time()

    all_results = []

    print("=" * 65)
    print("ORACLE - MULTI-SEED ACTIVE LEARNING EXPERIMENT")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Initial labeled: {INITIAL_LABELED}")
    print(f"Validation size: {VALIDATION_SIZE}")
    print(f"Query size: {QUERY_SIZE}")
    print(f"Maximum budget: {MAX_BUDGET}")
    print(f"Epochs: {EPOCHS}")

    # ----------------------------------------------
    # Run all strategy-seed combinations
    # ----------------------------------------------

    for strategy in STRATEGIES:

        for seed in SEEDS:

            run_results = run_experiment(
                strategy_name=strategy,
                seed=seed,
            )

            all_results.extend(
                run_results
            )

    # ----------------------------------------------
    # Save raw results
    # ----------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    raw_path = (
        EXPERIMENT_DIR
        / "multi_seed_metrics.csv"
    )

    results_df.to_csv(
        raw_path,
        index=False,
    )

    print(
        f"\nRaw results saved: {raw_path}"
    )

    # ----------------------------------------------
    # Aggregate validation results
    # ----------------------------------------------

    aggregate_df = (
        results_df
        .groupby(
            ["strategy", "budget"]
        )["validation_accuracy"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    aggregate_path = (
        EXPERIMENT_DIR
        / "multi_seed_aggregate.csv"
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    # ----------------------------------------------
    # Aggregate final test results
    # ----------------------------------------------

    test_df = results_df.dropna(
        subset=["test_accuracy"]
    )

    test_summary = (
        test_df
        .groupby("strategy")["test_accuracy"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )

    test_aggregate_path = (
        EXPERIMENT_DIR
        / "multi_seed_test_aggregate.csv"
    )

    test_summary.to_csv(
        test_aggregate_path,
        index=False,
    )

    test_json_path = (
        EXPERIMENT_DIR
        / "multi_seed_test_summary.json"
    )

    test_summary.to_json(
        test_json_path,
        orient="records",
        indent=4,
    )

    # ----------------------------------------------
    # Save experiment summary
    # ----------------------------------------------

    total_time = (
        time.time() - start_time
    )

    summary = {
        "dataset": "CIFAR-10",
        "seeds": SEEDS,
        "strategies": STRATEGIES,
        "initial_labeled": INITIAL_LABELED,
        "validation_size": VALIDATION_SIZE,
        "query_size": QUERY_SIZE,
        "max_budget": MAX_BUDGET,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": str(DEVICE),
        "runtime_seconds": total_time,
        "runtime_minutes": total_time / 60,
        "test_results": test_summary.to_dict(
            orient="records"
        ),
    }

    summary_path = (
        EXPERIMENT_DIR
        / "multi_seed_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ----------------------------------------------
    # Display results
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("MULTI-SEED EXPERIMENT COMPLETED")
    print("=" * 65)

    print("\nValidation Accuracy (Mean ± Std):")

    print(
        aggregate_df.to_string(
            index=False
        )
    )

    print("\nFinal Test Accuracy:")

    print(
        test_summary.to_string(
            index=False
        )
    )

    print(
        f"\nTotal Runtime: {total_time / 60:.2f} minutes"
    )

    print("\nGenerated files:")

    print(f"1. {raw_path}")
    print(f"2. {aggregate_path}")
    print(f"3. {test_aggregate_path}")
    print(f"4. {test_json_path}")
    print(f"5. {summary_path}")

    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()