
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.data.dataset import CIFAR10DataModule
from src.models.cnn import BaselineCNN


ROOT = Path(__file__).resolve().parents[1]
MODEL_LAB_DIR = ROOT / "experiments" / "model_lab"
CHECKPOINT_DIR = MODEL_LAB_DIR / "checkpoints"
HISTORY_PATH = MODEL_LAB_DIR / "history.json"

MODEL_LAB_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ModelLab:
    """Manages asynchronous CNN training and experiment records."""

    def __init__(self):
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None

        self.status: dict[str, Any] = {
            "running": False,
            "experiment_id": None,
            "state": "idle",
            "epoch": 0,
            "completed_epochs": 0,
            "total_epochs": 0,
            "train_loss": None,
            "train_accuracy": None,
            "validation_loss": None,
            "validation_accuracy": None,
            "best_validation_accuracy": None,
            "test_accuracy": None,
            "evaluation": None,
            "history": [],
            "error": None,
        }

    def _read_history(self) -> list[dict[str, Any]]:
        if not HISTORY_PATH.exists():
            return []

        try:
            with HISTORY_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)

            return data if isinstance(data, list) else []

        except (json.JSONDecodeError, OSError):
            return []

    def _write_history(self, history: list[dict[str, Any]]) -> None:
        temporary_path = HISTORY_PATH.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        temporary_path.replace(HISTORY_PATH)

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._read_history()

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        for experiment in self._read_history():
            if experiment.get("experiment_id") == experiment_id:
                return experiment

        return None

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.status)

    def start_training(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        initial_labeled_size: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:

        with self.lock:
            if self.status["running"]:
                raise RuntimeError(
                    "A training experiment is already running."
                )

            experiment_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid.uuid4().hex[:8]
            )

            config = {
                "model": "BaselineCNN",
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "initial_labeled_size": initial_labeled_size,
                "validation_size": 5000,
                "seed": seed,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
            }

            self.status = {
                "running": True,
                "experiment_id": experiment_id,
                "state": "queued",
                "epoch": 0,
                "completed_epochs": 0,
                "total_epochs": epochs,
                "train_loss": None,
                "train_accuracy": None,
                "validation_loss": None,
                "validation_accuracy": None,
                "best_validation_accuracy": None,
                "test_accuracy": None,
                "evaluation": None,
                "history": [],
                "error": None,
            }

            self.worker = threading.Thread(
                target=self._train,
                args=(experiment_id, config),
                daemon=True,
            )
            self.worker.start()

        return {
            "started": True,
            "experiment_id": experiment_id,
            "config": config,
        }

    def _update_status(self, **updates: Any) -> None:
        with self.lock:
            self.status.update(updates)

    @staticmethod
    def _evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:

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

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()

                total += batch_size

        if total == 0:
            return 0.0, 0.0

        return (
            total_loss / total,
            100.0 * correct / total,
        )

    @staticmethod
    def _evaluate_classification(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, Any]:

        model.eval()

        all_targets: list[int] = []
        all_predictions: list[int] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                all_targets.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

        label_ids = list(range(len(CLASS_NAMES)))

        matrix = confusion_matrix(
            all_targets,
            all_predictions,
            labels=label_ids,
        )

        report = classification_report(
            all_targets,
            all_predictions,
            labels=label_ids,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        per_class = {}

        for class_name in CLASS_NAMES:
            values = report[class_name]

            per_class[class_name] = {
                "precision": round(values["precision"] * 100, 4),
                "recall": round(values["recall"] * 100, 4),
                "f1_score": round(values["f1-score"] * 100, 4),
                "support": int(values["support"]),
            }

        macro = report["macro avg"]
        weighted = report["weighted avg"]

        return {
            "accuracy": round(
                accuracy_score(all_targets, all_predictions) * 100,
                4,
            ),
            "macro_avg": {
                "precision": round(macro["precision"] * 100, 4),
                "recall": round(macro["recall"] * 100, 4),
                "f1_score": round(macro["f1-score"] * 100, 4),
            },
            "weighted_avg": {
                "precision": round(weighted["precision"] * 100, 4),
                "recall": round(weighted["recall"] * 100, 4),
                "f1_score": round(weighted["f1-score"] * 100, 4),
            },
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "class_names": CLASS_NAMES,
            "test_samples": len(all_targets),
        }

    def _train(
        self,
        experiment_id: str,
        config: dict[str, Any],
    ) -> None:

        started_datetime = datetime.now(timezone.utc)
        started_at = started_datetime.isoformat()

        checkpoint_path = (
            CHECKPOINT_DIR / f"{experiment_id}_best.pth"
        )

        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "duration_seconds": None,
            "config": config,
            "history": [],
            "completed_epochs": 0,
            "best_epoch": None,
            "best_validation_accuracy": None,
            "test_accuracy": None,
            "evaluation": None,
            "checkpoint": None,
            "error": None,
        }

        try:
            self._update_status(state="preparing")

            torch.manual_seed(config["seed"])

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config["seed"])

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            data_module = CIFAR10DataModule(
                data_dir=str(ROOT / "data" / "raw"),
                batch_size=config["batch_size"],
                initial_labeled_size=config["initial_labeled_size"],
                validation_size=config["validation_size"],
                num_workers=0,
                seed=config["seed"],
            )

            data_module.prepare_data()
            data_module.setup()

            (
                train_loader,
                _,
                validation_loader,
                test_loader,
            ) = data_module.get_loaders()

            model = BaselineCNN(num_classes=10).to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
            )

            best_validation_accuracy = -1.0
            best_epoch = None

            self._update_status(state="training")

            for epoch in range(1, config["epochs"] + 1):
                model.train()

                running_loss = 0.0
                correct = 0
                total = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    batch_size = labels.size(0)

                    running_loss += loss.item() * batch_size

                    correct += (
                        outputs.argmax(dim=1) == labels
                    ).sum().item()

                    total += batch_size

                train_loss = running_loss / max(total, 1)
                train_accuracy = 100.0 * correct / max(total, 1)

                validation_loss, validation_accuracy = self._evaluate(
                    model,
                    validation_loader,
                    criterion,
                    device,
                )

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_accuracy": round(train_accuracy, 4),
                    "validation_loss": round(validation_loss, 6),
                    "validation_accuracy": round(
                        validation_accuracy,
                        4,
                    ),
                }

                record["history"].append(epoch_record)

                completed_epochs = len(record["history"])
                record["completed_epochs"] = completed_epochs

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_epoch = epoch

                    torch.save(
                        {
                            "experiment_id": experiment_id,
                            "model_name": "BaselineCNN",
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "best_validation_accuracy": (
                                best_validation_accuracy
                            ),
                            "epoch": epoch,
                        },
                        checkpoint_path,
                    )

                self._update_status(
                    state="training",
                    epoch=epoch,
                    completed_epochs=completed_epochs,
                    train_loss=round(train_loss, 6),
                    train_accuracy=round(train_accuracy, 4),
                    validation_loss=round(validation_loss, 6),
                    validation_accuracy=round(
                        validation_accuracy,
                        4,
                    ),
                    best_validation_accuracy=round(
                        best_validation_accuracy,
                        4,
                    ),
                    history=list(record["history"]),
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            model.load_state_dict(checkpoint["model_state_dict"])

            self._update_status(state="evaluating")

            _, test_accuracy = self._evaluate(
                model,
                test_loader,
                criterion,
                device,
            )

            evaluation = self._evaluate_classification(
                model,
                test_loader,
                device,
            )

            completed_datetime = datetime.now(timezone.utc)
            duration_seconds = round(
                (completed_datetime - started_datetime).total_seconds(),
                2,
            )

            record.update(
                {
                    "status": "completed",
                    "completed_at": completed_datetime.isoformat(),
                    "duration_seconds": duration_seconds,
                    "completed_epochs": len(record["history"]),
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": round(
                        best_validation_accuracy,
                        4,
                    ),
                    "test_accuracy": round(test_accuracy, 4),
                    "evaluation": evaluation,
                    "checkpoint": str(
                        checkpoint_path.relative_to(ROOT)
                    ),
                }
            )

            history = self._read_history()
            history.insert(0, record)
            self._write_history(history)

            self._update_status(
                running=False,
                state="completed",
                epoch=len(record["history"]),
                completed_epochs=len(record["history"]),
                total_epochs=config["epochs"],
                best_validation_accuracy=round(
                    best_validation_accuracy,
                    4,
                ),
                test_accuracy=round(test_accuracy, 4),
                evaluation=evaluation,
                error=None,
            )

        except Exception as exc:
            completed_datetime = datetime.now(timezone.utc)

            record.update(
                {
                    "status": "failed",
                    "completed_at": completed_datetime.isoformat(),
                    "duration_seconds": round(
                        (
                            completed_datetime - started_datetime
                        ).total_seconds(),
                        2,
                    ),
                    "completed_epochs": len(record["history"]),
                    "best_validation_accuracy": (
                        round(best_validation_accuracy, 4)
                        if "best_validation_accuracy" in locals()
                        and best_validation_accuracy >= 0
                        else None
                    ),
                    "best_epoch": (
                        best_epoch
                        if "best_epoch" in locals()
                        else None
                    ),
                    "error": str(exc),
                }
            )

            if checkpoint_path.exists():
                record["checkpoint"] = str(
                    checkpoint_path.relative_to(ROOT)
                )

            try:
                history = self._read_history()
                history.insert(0, record)
                self._write_history(history)
            except Exception:
                pass

            self._update_status(
                running=False,
                state="failed",
                completed_epochs=len(record["history"]),
                error=str(exc),
            )


model_lab = ModelLab()