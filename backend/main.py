
from __future__ import annotations

import base64
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import torchvision
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.model_lab import model_lab


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "annotations.sqlite"
EXPERIMENTS_DIR = ROOT / "experiments"

CLASSES = [
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

app = FastAPI(title="ORACLE API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
        r"|^https://[a-zA-Z0-9-]+\.vercel\.app$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_dataset = None


def get_dataset():
    global _dataset

    if _dataset is None:
        _dataset = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR),
            train=True,
            download=True,
            transform=None,
        )

    return _dataset


def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_index INTEGER NOT NULL UNIQUE,
            chosen_label TEXT NOT NULL,
            revealed_label TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    return conn


def _read_csv(filename: str) -> list[dict[str, Any]]:
    path = EXPERIMENTS_DIR / filename

    if not path.exists():
        return []

    try:
        frame = pd.read_csv(path)
        frame = frame.astype(object).where(pd.notna(frame), None)
        return frame.to_dict(orient="records")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {filename}: {exc}",
        ) from exc


class AnnotationIn(BaseModel):
    image_index: int = Field(ge=0, lt=50000)
    chosen_label: str
    reveal_truth: bool = False


class TrainingConfig(BaseModel):
    epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        le=0.1,
    )
    initial_labeled_size: int = Field(
        default=1000,
        ge=100,
        le=45000,
    )
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "ORACLE API",
    }


@app.get("/api/dataset/summary")
def summary():
    return {
        "name": "CIFAR-10",
        "total_train": 50000,
        "total_test": 10000,
        "classes": CLASSES,
        "image_shape": [32, 32, 3],
    }


@app.get("/api/experiments/metrics")
def experiment_metrics():
    filenames = [
        "multi_seed_metrics.csv",
        "multi_seed_aggregate.csv",
        "multi_seed_test_summary.json",
        "multi_seed_test_aggregate.csv",
        "multi_seed_summary.json",
    ]

    result: dict[str, Any] = {
        "source": "experiments/",
        "files": {},
        "missing_files": [],
    }

    for filename in filenames:
        path = EXPERIMENTS_DIR / filename

        if not path.exists():
            result["files"][filename] = None
            result["missing_files"].append(filename)
            continue

        try:
            if path.suffix.lower() == ".csv":
                result["files"][filename] = _read_csv(filename)
            else:
                with path.open("r", encoding="utf-8") as file:
                    result["files"][filename] = json.load(file)

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read {filename}: {exc}",
            ) from exc

    result["available"] = (
        len(result["missing_files"]) < len(filenames)
    )

    return result


@app.get("/api/images/{image_index}")
def get_image(image_index: int):
    if image_index < 0 or image_index >= 50000:
        raise HTTPException(404, "Image index out of range")

    ds = get_dataset()
    image, _ = ds[image_index]

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    return {
        "image_index": image_index,
        "image_base64": base64.b64encode(
            buf.getvalue()
        ).decode("ascii"),
    }


@app.get("/api/annotations")
def list_annotations():
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM annotations ORDER BY id DESC LIMIT 500"
        ).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) FROM annotations"
        ).fetchone()[0]

    return {
        "total": total,
        "items": [dict(row) for row in rows],
    }


@app.post("/api/annotations")
def save_annotation(payload: AnnotationIn):
    if payload.chosen_label not in CLASSES:
        raise HTTPException(400, "Unknown class label")

    ds = get_dataset()
    _, true_index = ds[payload.image_index]

    truth = (
        CLASSES[true_index]
        if payload.reveal_truth
        else None
    )

    with db() as conn:
        conn.execute(
            """
            INSERT INTO annotations (
                image_index,
                chosen_label,
                revealed_label
            )
            VALUES (?, ?, ?)
            ON CONFLICT(image_index) DO UPDATE SET
                chosen_label = excluded.chosen_label,
                revealed_label = excluded.revealed_label,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                payload.image_index,
                payload.chosen_label,
                truth,
            ),
        )

        conn.commit()

    return {
        "saved": True,
        "image_index": payload.image_index,
        "chosen_label": payload.chosen_label,
        "revealed_label": truth,
        "matches_truth": (
            payload.chosen_label == truth
        ) if truth else None,
    }


@app.delete("/api/annotations/{image_index}")
def delete_annotation(image_index: int):
    with db() as conn:
        cur = conn.execute(
            "DELETE FROM annotations WHERE image_index = ?",
            (image_index,),
        )

        conn.commit()

    if cur.rowcount == 0:
        raise HTTPException(404, "Annotation not found")

    return {
        "deleted": True,
        "image_index": image_index,
    }


# --------------------------------------------------
# MODEL LAB API
# --------------------------------------------------

@app.post("/api/model-lab/train", status_code=202)
def start_model_training(payload: TrainingConfig):
    try:
        return model_lab.start_training(
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
            initial_labeled_size=payload.initial_labeled_size,
            seed=payload.seed,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.get("/api/model-lab/status")
def model_lab_status():
    return model_lab.get_status()


@app.get("/api/model-lab/experiments")
def model_lab_experiments():
    experiments = model_lab.list_experiments()

    return {
        "total": len(experiments),
        "items": experiments,
    }


@app.get("/api/model-lab/experiments/{experiment_id}")
def model_lab_experiment_detail(experiment_id: str):
    experiment = model_lab.get_experiment(experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return experiment