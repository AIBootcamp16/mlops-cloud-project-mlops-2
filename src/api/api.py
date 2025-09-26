from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..model.model import Recommender

app = FastAPI(title="Spotify-style Recommender API")

MODEL_DIR = "./models"

class RecommendRequest(BaseModel):
    by: str = "track_name"
    query: str
    top_k: int = 10

@app.on_event("startup")
def _load_model():
    global rec
    rec = Recommender.load(MODEL_DIR)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        df = rec.recommend(by=req.by, query=req.query, top_k=req.top_k)
        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
