
"""
ORACLE - Uncertainty Sampling Experiment

Evaluates model performance as the labeled dataset
grows through least-confidence uncertainty sampling.
"""

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data.dataset import CIFAR10DataModule
from src.models.cnn import BaselineCNN
from src.active_learning.uncertainty_sampler import UncertaintySampler


def train_model(model, loader, criterion, optimizer, device, epochs):
    """Train a model using the currently labeled samples."""

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model without updating its weights."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total,
    )


def main():

    seed = 42
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Experiment device: {device}")

    # Experiment settings
    initial_labeled_size = 1000
    query_size = 500
    max_labeled_size = 5000
    epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Prepare directories
    Path("experiments").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # Load dataset and fixed splits
    data_module = CIFAR10DataModule(
        batch_size=batch_size,
        initial_labeled_size=initial_labeled_size,
        validation_size=5000,
        seed=seed,
    )

    data_module.prepare_data()
    data_module.setup()

    train_dataset = data_module.train_dataset
    validation_dataset = data_module.validation_dataset
    test_dataset = data_module.test_dataset

    # Recover original dataset indices
    labeled_indices = list(
        data_module.labeled_dataset.indices
    )

    unlabeled_indices = list(
        data_module.unlabeled_dataset.indices
    )

    # Uncertainty sampler manages only the unlabeled pool
    sampler = UncertaintySampler(
        pool_indices=unlabeled_indices,
    )

    criterion = nn.CrossEntropyLoss()

    results = []
    iteration = 0

    best_val_accuracy = 0.0
    best_model_state = None

    experiment_start = time.time()

    print("\n--- Uncertainty Sampling Experiment ---")

    while len(labeled_indices) <= max_labeled_size:

        iteration += 1
        current_budget = len(labeled_indices)

        print(
            f"\nIteration {iteration} | "
            f"Labeled samples: {current_budget}"
        )

        iteration_start = time.time()

        # Build training loader from selected labeled samples
        labeled_subset = Subset(
            train_dataset,
            labeled_indices,
        )

        train_loader = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Train a fresh model for each budget
        model = BaselineCNN(num_classes=10).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        train_loss, train_accuracy = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epochs,
        )

        val_loss, val_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        iteration_time = time.time() - iteration_start

        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Iteration Time: {iteration_time:.2f} seconds")

        results.append({
            "iteration": iteration,
            "labeled_samples": current_budget,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 2),
            "validation_loss": round(val_loss, 4),
            "validation_accuracy": round(val_accuracy, 2),
            "iteration_time_seconds": round(iteration_time, 2),
        })

        # Track best validation checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        # Save progress after every iteration
        with open(
            "experiments/uncertainty_sampling_metrics.csv",
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys(),
            )
            writer.writeheader()
            writer.writerows(results)

        if current_budget == max_labeled_size:
            break

        # Query new samples using current model
        next_query_size = min(
            query_size,
            max_labeled_size - current_budget,
        )

        print(
            f"Selecting {next_query_size} uncertain samples..."
        )

        newly_selected = sampler.query(
            model=model,
            dataset=train_dataset,
            n_samples=next_query_size,
            device=device,
            batch_size=batch_size,
        )

        labeled_indices.extend(newly_selected)

        print(
            f"Selected {len(newly_selected)} "
            "additional uncertain samples."
        )

    total_time = time.time() - experiment_start

    # Final test evaluation using the best validation checkpoint
    final_model = BaselineCNN(num_classes=10).to(device)
    final_model.load_state_dict(best_model_state)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loss, test_accuracy = evaluate(
        final_model,
        test_loader,
        criterion,
        device,
    )

    torch.save(
        best_model_state,
        "models/uncertainty_sampling_best.pth",
    )

    summary = {
        "algorithm": "Least Confidence Uncertainty Sampling",
        "initial_labeled_size": initial_labeled_size,
        "query_size": query_size,
        "maximum_labeled_size": max_labeled_size,
        "epochs_per_iteration": epochs,
        "best_validation_accuracy": round(
            best_val_accuracy, 2
        ),
        "final_test_loss": round(test_loss, 4),
        "final_test_accuracy": round(test_accuracy, 2),
        "total_training_time_seconds": round(total_time, 2),
        "best_model_path": "models/uncertainty_sampling_best.pth",
    }

    with open(
        "experiments/uncertainty_sampling_summary.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n--- Uncertainty Sampling Experiment Complete ---")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Total time: {total_time:.2f} seconds")
    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()