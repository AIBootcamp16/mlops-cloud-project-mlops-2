# ============================================================
# 🎧 Spotify Music Recommender API (Production + CI/CD Safe)
# ------------------------------------------------------------
# 기능 요약:
#  - Spotify API 인증 및 트랙 이미지 URL 연동
#  - Finder, FAISS, LGBM, TopN 모델 로드 및 예외 처리
#  - CI/CD 환경에서 dataset 누락 시에도 부팅 유지
# ------------------------------------------------------------
# 작성자: gogoAiHunters Team
# ============================================================

import joblib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..model.NModel import LGBMRecommender, FAISSRecommender, Finder
from ..model.logger import RecoLogger, RecommendLog 
from ..model.TopN import TopN_Model  # TopN 모델 로직 임포트

import numpy as np
import time
import pickle
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

# ------------------------------------------------------------
# 1️⃣ FastAPI 인스턴스
# ------------------------------------------------------------
app = FastAPI(title="Spotify Recommender API")

# ------------------------------------------------------------
# 2️⃣ Spotify 인증 (서버 시작 시 1회 수행)
# ------------------------------------------------------------
sp = None
SPOTIPY_ERROR = None

@app.on_event("startup")
def setup_spotify_auth():
    """서버 시작 시 Spotify 인증 시도"""
    global sp, SPOTIPY_ERROR

    CLIENT_ID = os.environ.get('CLIENT_ID')
    CLIENT_SECRET = os.environ.get('CLIENT_SECRET')

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


# ------------------------------------------------------------
# 3️⃣ 요청 스키마 정의
# ------------------------------------------------------------
class SearchRequest(BaseModel):
    by: str = "track_name"
    query: str
    limit: int = 50


class RecommendRankedRequest(BaseModel):
    by: str
    query: str
    top_k: int | None = 10


# ------------------------------------------------------------
# 4️⃣ 경로 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = BASE_DIR / "models" / "mappings.pkl"
FAISS_PATH = BASE_DIR / "models" / "faiss.index"

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
]


# ------------------------------------------------------------
# 5️⃣ 서버 시작 시 아티팩트 로드
# ------------------------------------------------------------
@app.on_event("startup")
def _load_artifacts():
    """Finder, 모델, 데이터셋 등 로드"""
    print("🚀 [Startup] Loading artifacts...")

    # (1) Finder 로드
    try:
        app.state.fin = Finder.load(str(MODEL_DIR))
    except FileNotFoundError as e:
        print(f"⚠️ Warning: Finder artifacts not found. ERROR: {e}") 
        app.state.fin = None
    except Exception as e:
        print(f"❌ CRITICAL ERROR during Finder load: {e}")
        app.state.fin = None

    # (2) 데이터 로드 (CI/CD 안전 모드)
    try:
        if not DATA_PATH.exists():
            print(f"⚠️ DATA not found: {DATA_PATH}. Running API without dataset (CI/CD mode).")
            app.state.df = None  # 데이터 없이도 FastAPI 부팅 유지
        else:
            app.state.df = pd.read_csv(DATA_PATH)
            print(f"✅ Dataset loaded successfully: {len(app.state.df)} records.")
            if "image_url" not in app.state.df.columns:
                print("⚠️ Warning: 'image_url' column not found in dataset.")
            if "track_id" not in app.state.df.columns:
                print("⚠️ Warning: dataset missing 'track_id' column.")
    except Exception as e:
        print(f"❌ Dataset load failed: {e}")
        app.state.df = None

    # (3) FAISS 추천기 로드
    try:
        app.state.faiss_rec = FAISSRecommender.load(str(MODEL_DIR))
    except FileNotFoundError:
        print("⚠️ FAISS recommender not found (optional).")
        app.state.faiss_rec = None
    except Exception as e:
        print(f"❌ Failed to load FAISS recommender: {e}")
        app.state.faiss_rec = None

    # (4) LGBM 모델 로드
    try:
        app.state.lgbm_rec = LGBMRecommender.load(str(MODEL_DIR))
    except FileNotFoundError:
        print("⚠️ LGBM recommender not found (optional).")
        app.state.lgbm_rec = None
    except Exception as e:
        print(f"❌ Failed to load LGBM recommender: {e}")
        app.state.lgbm_rec = None

    # (5) TopN 모델 초기화
    try:
        app.state.topn_model = TopN_Model(
            finder=app.state.fin,
            faiss_recommender=app.state.faiss_rec,
            lgbm_recommender=app.state.lgbm_rec,
            data_df=app.state.df
        )
        print("✅ TopN Model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize TopN model: {e}")
        app.state.topn_model = None

    print("[Startup Completed] Models and Data initialized.")


# ------------------------------------------------------------
# 6️⃣ Spotify 이미지 URL 보강
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 7️⃣ Healthcheck
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Server is running 🚀"}


# ------------------------------------------------------------
# 8️⃣ Search Endpoint
# ------------------------------------------------------------
@app.post("/search")
def search(req: SearchRequest):
    if app.state.fin is None:
        raise HTTPException(status_code=500, detail="Finder not initialized.")

    try:
        matches = app.state.fin._lookup_indices(by=req.by, query=req.query, max_matches=req.limit)
        idx = app.state.fin.artifacts.id_index
        cols = [c for c in ["track_id", "track_name", "artist_name", "image_url"] if c in idx.columns]
        df = idx.iloc[matches][cols].reset_index(drop=True)

        if "image_url" not in df.columns or df["image_url"].isnull().all():
            df = enrich_with_image_url(df)

        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------
# 9️⃣ Recommend Endpoint
# ------------------------------------------------------------
@app.post("/recommend_ranked")
def recommend(req: RecommendRankedRequest):
    if app.state.topn_model is None:
        raise HTTPException(status_code=500, detail="TopN model not initialized.")

    try:
        t0 = time.time()
        top_k_limit = req.top_k or 10
        TopN_result = app.state.topn_model.Search(by=req.by, query=req.query, top_k=top_k_limit)
        elapsed = time.time() - t0

        if "image_url" not in TopN_result.columns or TopN_result["image_url"].isnull().all():
            TopN_result = enrich_with_image_url(TopN_result)

        return {"items": TopN_result.to_dict(orient="records"), "elapsed_time": elapsed}
    except Exception as e:
        print(f"❌ Error in /recommend_ranked: {e}") 
        raise HTTPException(status_code=400, detail=f"Recommendation failed: {str(e)}")


# ------------------------------------------------------------
# 🔟 로컬 실행 (개발 전용)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)
