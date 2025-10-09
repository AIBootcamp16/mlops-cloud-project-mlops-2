from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from ..model.NModel import LGBMRecommender, FAISSRecommender, Finder
from ..model.logger import RecoLogger, RecommendLog 
from ..model.TopN import TopN_Model

import os
import time
import pickle
import numpy as np
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

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

# ======= start up =======
sp = None
SPOTIPY_ERROR = None

@app.on_event("startup")
def setup_spotify_auth():
    """서버 시작 시 Spotify 인증 시도"""
    global sp, SPOTIPY_ERROR

    CLIENT_ID = os.environ.get('CLIENT_ID')
    CLIENT_SECRET = os.environ.get('CLIENT_SECRET')

    print("##")
    print(f"CLIENT_ID: {CLIENT_ID}")

    try:
        auth_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        SPOTIPY_ERROR = None
        print("✅ Spotify 인증 성공. FastAPI 서버 가동.")
    except Exception as e:
        print(f"❌ Spotify 인증 실패. 오류 메시지: {e}")
        sp = None
        SPOTIPY_ERROR = str(e)

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

# ======= Spotify 이미지 URL =======
def enrich_with_image_url(df: pd.DataFrame) -> pd.DataFrame:
    """track_id 기준으로 Spotify API에서 image_url 가져오기"""
    if sp is None or SPOTIPY_ERROR:
        print(f"⚠️ Spotify API 비활성: {SPOTIPY_ERROR}")
        df["image_url"] = None
        return df

    if df.empty or "track_id" not in df.columns:
        df["image_url"] = None
        return df

    track_ids = df["track_id"].tolist()
    records = [None] * len(track_ids)

    for i in range(0, len(track_ids), 50):
        chunk_ids = track_ids[i:i+50]
        try:
            track_details = sp.tracks(chunk_ids)
            if track_details and track_details.get('tracks'):
                for j, t in enumerate(track_details['tracks']):
                    if t and t.get('album') and t['album'].get('images'):
                        records[i + j] = t['album']['images'][0]['url']
        except requests.exceptions.HTTPError as e:
            print(f"❌ Spotify API HTTP 오류: {e.response.status_code}")
        except Exception as e:
            print(f"❌ Spotify API 호출 오류: {e}")
            continue

    df["image_url"] = records
    return df

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

        if "image_url" not in df.columns or df["image_url"].isnull().all():
            df = enrich_with_image_url(df)

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
        
        if "image_url" not in TopN_result.columns or TopN_result["image_url"].isnull().all():
            TopN_result = enrich_with_image_url(TopN_result)
        
        seed_ids = [req.query]                      # 시드 1개라고 가정
        returned_ids = list(TopN_result["track_id"]) 
        
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

    