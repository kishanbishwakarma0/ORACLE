# ORACLE milestone 1 — API foundation

This adds a real local API for CIFAR-10 image retrieval and persistent annotation records.

## Install (from project root)
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn pillow
```
Ensure the project's existing `torch` and `torchvision` installations are available in this environment.

## Start API (from project root)
```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs to inspect the endpoints.

## Endpoints
- `GET /api/health` — API health
- `GET /api/dataset/summary` — dataset metadata
- `GET /api/images/{image_index}` — image bytes as base64; does not expose its label
- `GET /api/annotations` — saved annotations
- `POST /api/annotations` — save chosen class; optional `reveal_truth` for simulated-oracle evaluation
- `DELETE /api/annotations/{image_index}` — remove a saved annotation

The first dataset request may download CIFAR-10. Annotation records are stored in `data/annotations.sqlite`.
