from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..model.model import Recommender
from ..model.logger import RecoLogger, RecommendLog 

app = FastAPI(title="Spotify-style Recommender API")

MODEL_DIR = "./models"

class RecommendRequest(BaseModel):
    by: str = "track_name"
    query: str
    top_k: int = 10

class SearchRequest(BaseModel):
    by: str = "track_name"
    query: str
    limit: int = 50

@app.on_event("startup")
def _load_model():
    global rec
    rec = Recommender.load(MODEL_DIR)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search(req: SearchRequest):
    try:
        matches = rec._lookup_indices(by=req.by, query=req.query, max_matches=req.limit)
        idx = rec.artifacts.id_index
        cols = [c for c in ["track_id", "track_name", "artist_name"] if c in idx.columns]
        df = idx.iloc[matches][cols].reset_index(drop=True)
        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/recommend")
def recommend(req: RecommendRequest):
    t0 = time.time()
    try:
        df = rec.recommend(by=req.by, query=req.query, top_k=req.top_k)
        elapsed = time.time() - t0

        seed_ids = rec.artifacts.id_index.iloc[
            rec._lookup_indices(by=req.by, query=req.query)
        ]["track_id"].tolist()[:10]

        logger = RecoLogger()
        logger.log_recommend(RecommendLog(
            by_field=req.by,
            query=req.query,
            top_k=req.top_k,
            elapsed_sec=elapsed,
            seed_track_ids=seed_ids,
            returned_track_ids=df["track_id"].tolist()
        ))

        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
