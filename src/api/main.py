import sys
sys.path.append("../../model")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import time, os, json

from spotify_recommender import Recommender

MODEL_DIR = os.getenv("MODEL_DIR", "../model")

app = FastAPI(title="Music Recommender API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecRequest(BaseModel):
    by: Literal["track_name","artist_name","track_id"]
    query: str = Field(min_length=1)
    top_k: int = Field(ge=1, le=50)

class TrackItem(BaseModel):
    id: str
    name: str
    artist: str
    score: Optional[float] = None

class RecResponse(BaseModel):
    items: List[TrackItem]
    elapsed_sec: float

# 서버 시작 시 한 번만 로드 (핵심!)
@app.on_event("startup")
def _load_model_once():
    app.state.rec = Recommender.load(MODEL_DIR)

@app.post("/recommend", response_model=RecResponse)
def recommend(req: RecRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is empty")

    t0 = time.time()
    try:
        # ⬇️ 네 메서드 그대로 호출
        df = app.state.rec.recommend(by=req.by, query=req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"model error: {e}")

    # DF → JSON (컬럼명이 다를 수 있으니 안전 매핑)
    items = []
    for _, row in df.iterrows():
        items.append({
            "id": str(row.get("track_id") or row.get("id") or ""),
            "name": str(row.get("track_name") or row.get("name") or ""),
            "artist": str(row.get("artist_name") or row.get("artist") or ""),
            "score": float(row["score"]) if "score" in row and row["score"] is not None else None,
        })

    return {"items": items, "elapsed_sec": time.time() - t0}
