from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from ..model.NModel import LGBMRecommender, FAISSRecommender, Finder
from ..model.logger import RecoLogger, RecommendLog 
from ..model.TopN import TopN_Model

import time
import pickle
import pandas as pd

reco_logger = RecoLogger()
myTopN_Model = TopN_Model()
app = FastAPI()

# ======= 요청 스키마 =======
class SearchRequest(BaseModel):
    by: str = "track_name"
    query: str
    limit: int = 50

class RecommendRankedRequest(BaseModel):
    by: str            # "track_name" / "artist" 등
    query: str         # 검색어
    top_k: int | None = 10


# ======= 경로 설정 =======
BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = BASE_DIR / "models" / "mappings.pkl"
FAISS_PATH = BASE_DIR / "models" / "faiss.index"   

FEATURE_COLS = [
    "danceability","energy","key","loudness","mode","speechiness",
    "acousticness","instrumentalness","liveness","valence","tempo","duration_ms"
]

# ======= 서버 시작 시 한 번만 로드 =======
@app.on_event("startup")
def _load_artifacts():
    app.state.fin = Finder.load(str(MODEL_DIR))

    # 1) 데이터
    if not DATA_PATH.exists():
        raise RuntimeError(f"DATA not found: {DATA_PATH}")
    app.state.df = pd.read_csv(DATA_PATH)
    if "track_id" not in app.state.df.columns:
        raise RuntimeError("dataset must contain 'track_id' column")

    # 3) FAISS 추천기 로드
    try:
        app.state.faiss_rec = FAISSRecommender.load(str(MODEL_DIR))
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS recommender: {e}")

    print("[startup] Artifacts loaded:",
          f"\n- df shape: {app.state.df.shape}",
          f"\n- faiss: {type(app.state.faiss_rec)}")

@app.get("/health")
def health():
    return {
        "status": "ok",
    }

# 음악 검색
@app.post("/search")
def search(req: SearchRequest):
    try:
        matches = app.state.fin._lookup_indices(by=req.by, query=req.query, max_matches=req.limit)
        idx = app.state.fin.artifacts.id_index
        cols = [c for c in ["track_id", "track_name", "artist_name"] if c in idx.columns]
        df = idx.iloc[matches][cols].reset_index(drop=True)
        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 음악 추천
@app.post("/recommend_ranked")
def recommend(req: RecommendRankedRequest):
    try:
        t0 = time.time()
        TopN_result = myTopN_Model.Search(by=req.by, query=req.query, top_k=10)
        elapsed = time.time() - t0
        
        seed_ids = [req.query]                      # 시드 1개라고 가정
        returned_ids = [rec for rec in TopN_result]  # 모델 결과
        
        reco_logger.log_recommend(
            RecommendLog(
                by_field="track_id",
                query=str(req.query),
                top_k=len(returned_ids),
                elapsed_sec=float(elapsed),
                seed_track_ids=seed_ids,
                returned_track_ids=returned_ids,
            )
        )
        return {"items": TopN_result.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    