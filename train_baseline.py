
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import CIFAR10DataModule
from src.models.cnn import BaselineCNN


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation or test data."""

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

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def main():

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    # Experiment configuration
    config = {
        "experiment_name": "baseline_cnn",
        "dataset": "CIFAR-10",
        "initial_labeled_size": 1000,
        "validation_size": 5000,
        "unlabeled_size": 44000,
        "test_size": 10000,
        "batch_size": 64,
        "epochs": 10,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "seed": seed,
        "device": str(device),
    }

    # Prepare directories
    Path("models").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)

    # Prepare dataset
    data_module = CIFAR10DataModule(
        batch_size=config["batch_size"],
        initial_labeled_size=config["initial_labeled_size"],
        validation_size=config["validation_size"],
        seed=seed,
    )

    data_module.prepare_data()
    data_module.setup()

    (
        train_loader,
        unlabeled_loader,
        validation_loader,
        test_loader,
    ) = data_module.get_loaders()

    # Initialize model
    model = BaselineCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    epochs = config["epochs"]
    best_val_accuracy = 0.0
    best_epoch = 0

    model_path = Path("models/baseline_cnn.pth")
    csv_path = Path("experiments/baseline_metrics.csv")
    config_path = Path("experiments/baseline_config.json")

    # Save experiment configuration
    with open(config_path, "w") as file:
        json.dump(config, file, indent=4)

    # CSV logging
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_time_seconds",
    ]

    start_time = time.time()

    with open(csv_path, "w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for epoch in range(epochs):

            epoch_start = time.time()

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:

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

            # Validation evaluation
            val_loss, val_accuracy = evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}%"
            )

            # Save epoch metrics
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_time_seconds": round(epoch_time, 2),
            })

            csv_file.flush()

            # Save best model using validation accuracy
            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    model_path,
                )

                print("Best validation model saved.")

    total_time = time.time() - start_time

    # Load best validation checkpoint
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    # Final test evaluation (only once)
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # Save final summary
    summary = {
        **config,
        "best_epoch": best_epoch,
        "best_validation_accuracy": round(
            best_val_accuracy, 2
        ),
        "final_test_loss": round(test_loss, 4),
        "final_test_accuracy": round(test_accuracy, 2),
        "training_time_seconds": round(total_time, 2),
        "model_path": str(model_path),
    }

    summary_path = Path("experiments/baseline_summary.json")

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=4)

    print("\n--- ORACLE Baseline Experiment Complete ---")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Training time: {total_time:.2f} seconds")

    print("\nSaved artifacts:")
    print(f"Model: {model_path}")
    print(f"Metrics: {csv_path}")
    print(f"Configuration: {config_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()