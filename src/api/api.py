import joblib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# LGBMRecommender, FAISSRecommender, Finder는 모델 로드에 사용
from ..model.NModel import LGBMRecommender, FAISSRecommender, Finder
from ..model.logger import RecoLogger, RecommendLog 
from ..model.TopN import TopN_Model # TopN 모델 로직 임포트

import numpy as np
import time
import pickle
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests # requests 라이브러리도 필요합니다.

# FastAPI 앱 인스턴스 초기화
app = FastAPI(title="Spotify Recommender API")

# 제거했던 인증 코드를 여기에 넣어, 'app' 객체 정의 이후에 실행되도록 합니다.
# 필요한 전역 변수를 다시 정의합니다.
sp = None
SPOTIPY_ERROR = None

@app.on_event("startup")
def setup_spotify_auth():
    """서버 시작 시 Spotify 인증을 시도하고 결과를 전역 변수에 저장합니다."""
    global sp, SPOTIPY_ERROR
    
    # 환경 변수를 함수 내부에서 로드 (main scope의 CLIENT_ID/SECRET은 제거되었으므로)
    CLIENT_ID = os.environ.get('CLIENT_ID')
    CLIENT_SECRET = os.environ.get('CLIENT_SECRET')

    try:
        # 이 부분이 오류가 나도 'app' 객체는 이미 정의되었으므로 Uvicorn은 서버를 시작합니다.
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
        
# ======= 요청 스키마 =======
class SearchRequest(BaseModel):
    by: str = "track_name"
    query: str
    limit: int = 50

class RecommendRankedRequest(BaseModel):
    by: str              # "track_name" / "artist" 등
    query: str           # 검색어
    top_k: int | None = 10

# ======= 경로 설정 =======
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = BASE_DIR / "models" / "mappings.pkl" # Finder/Mappings 경로
FAISS_PATH = BASE_DIR / "models" / "faiss.index"   # FAISS 인덱스 경로

FEATURE_COLS = [
    "danceability","energy","key","loudness","mode","speechiness",
    "acousticness","instrumentalness","liveness","valence","tempo","duration_ms"
]

# ======= 서버 시작 시 한 번만 로드 =======
@app.on_event("startup")
def _load_artifacts():
    # 1) Finder 로드
    try:
        # Finder는 mappings.pkl 등 여러 아티팩트를 로드할 수 있음
        app.state.fin = Finder.load(str(MODEL_DIR))
    except FileNotFoundError as e:
        print(f"⚠️ Warning: Finder artifacts not found. ERROR: {e}") 
        app.state.fin = None
    except Exception as e:
        # 💡 이 부분이 중요합니다. 상세 에러 메시지 출력
        print(f"❌ CRITICAL ERROR during Finder load: {e}")
        app.state.fin = None # 오류 발생 시 None으로 설정
         
    # 2) 데이터 로드 (이 파일은 항상 존재해야 함)
    if not DATA_PATH.exists():
        # 데이터 파일이 없으면 API를 실행할 수 없음
        raise RuntimeError(f"DATA not found: {DATA_PATH}")
    app.state.df = pd.read_csv(DATA_PATH)
    # image_url 컬럼이 있는지 확인 (Streamlit에서 사용될 예정)
    if "image_url" not in app.state.df.columns:
        print("⚠️ Warning: 'image_url' column not found in dataset.")
    if "track_id" not in app.state.df.columns:
        raise RuntimeError("dataset must contain 'track_id' column")

    # 3) FAISS 추천기 로드
    try:
        app.state.faiss_rec = FAISSRecommender.load(str(MODEL_DIR))
    except FileNotFoundError as e:
        print(f"⚠️ Warning: FAISS recommender not found. Prediction endpoint may fail.")
        app.state.faiss_rec = None
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS recommender: {e}")
        
    # 4) LGBM 모델 로드
    try:
        app.state.lgbm_rec = LGBMRecommender.load(str(MODEL_DIR))
    except FileNotFoundError as e:
        print(f"⚠️ Warning: LGBM recommender not found (optional).")
        app.state.lgbm_rec = None
    except Exception as e:
        print(f"❌ Error loading LGBM recommender: {e}")
        app.state.lgbm_rec = None
        
    # 5) TopN 모델 초기화
    # 로드된 아티팩트가 없을 경우에도 TopN 모델이 초기화되도록 None을 전달
    try:
        app.state.topn_model = TopN_Model(
            finder=app.state.fin, 
            faiss_recommender=app.state.faiss_rec, 
            lgbm_recommender=app.state.lgbm_rec, 
            data_df=app.state.df
        )
    except Exception as e:
        # TopN 모델 초기화 실패 시 (의존성 문제 등)
        print(f"❌ ERROR: Failed to initialize TopN Model: {e}")
        # raise RuntimeError(...) # 강제 종료 방지를 위해 주석 처리하거나 제거
        app.state.topn_model = None
    
    # 로드 결과 로그 출력
    print("[startup] Artifacts loaded:",
          f"\n- df shape: {app.state.df.shape}",
          f"\n- Finder: {type(app.state.fin) if app.state.fin else 'None'}",
          f"\n- FAISS: {type(app.state.faiss_rec) if app.state.faiss_rec else 'None'}",
          f"\n- LGBM: {type(app.state.lgbm_rec) if app.state.lgbm_rec else 'None'}",
          f"\n- TopN Model Initialized: {type(app.state.topn_model)}")

# ----------------- [새로 추가된 부분] -----------------
def enrich_with_image_url(df: pd.DataFrame) -> pd.DataFrame:
    """
    track_id를 사용하여 Spotify에서 image_url을 가져와 DataFrame에 추가합니다.
    """
    if sp is None or SPOTIPY_ERROR:
        print(f"⚠️ Spotify API 비활성: {SPOTIPY_ERROR}")
        df["image_url"] = None
        return df

    if df.empty or "track_id" not in df.columns:
        df["image_url"] = None
        return df

    # track_id 리스트 추출
    track_ids = df["track_id"].tolist()
    records = [None] * len(track_ids)
    
    # Spotify API는 한 번에 최대 50개의 track_id만 처리할 수 있습니다.
    for i in range(0, len(track_ids), 50):
        chunk_ids = track_ids[i:i+50]
        try:
            track_details = sp.tracks(chunk_ids)
            if track_details and track_details.get('tracks'):
                for j, t in enumerate(track_details['tracks']):
                    if t and t.get('album') and t['album'].get('images'):
                        # 가장 큰 이미지 URL (0번째)을 저장
                        records[i + j] = t['album']['images'][0]['url']
        except requests.exceptions.HTTPError as e:
             # Spotify API 인증 오류 발생 시 (토큰 만료 등)
            print(f"❌ Spotify API HTTP 오류: {e.response.status_code}")
        except Exception as e:
            print(f"❌ Spotify API 호출 중 오류 발생: {e}")
            # 일부 오류가 나더라도 다음 청크로 진행

    df["image_url"] = records
    return df
# ----------------- [추가 끝] -----------------


@app.get("/health")
def health():
    return {
        "status": "ok",
    }

# 음악 검색
@app.post("/search")
def search(req: SearchRequest):
    if app.state.fin is None:
        raise HTTPException(status_code=500, detail="Finder is not initialized (missing model artifacts).")
        
    try:
        # 검색 결과에도 image_url 포함
        matches = app.state.fin._lookup_indices(by=req.by, query=req.query, max_matches=req.limit)
        idx = app.state.fin.artifacts.id_index
        # image_url 컬럼을 추가
        cols = [c for c in ["track_id", "track_name", "artist_name", "image_url"] if c in idx.columns]
        df = idx.iloc[matches][cols].reset_index(drop=True)

        # ----------------- [수정된 부분] -----------------
        # 데이터 파일에 image_url이 없거나 비어있는 경우, Spotify API로 채워 넣습니다.
        if "image_url" not in df.columns or df["image_url"].isnull().all():
            df = enrich_with_image_url(df)
        # ----------------- [수정 끝] -----------------

        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 음악 추천
@app.post("/recommend_ranked")
def recommend(req: RecommendRankedRequest):
    if app.state.topn_model is None:
        raise HTTPException(status_code=500, detail="TopN Model is not initialized (missing model artifacts).")
        
    try:
        t0 = time.time()
        top_k_limit = req.top_k if req.top_k is not None else 10
        
        TopN_result = app.state.topn_model.Search(
            by=req.by, 
            query=req.query, 
            top_k=top_k_limit
        )
        elapsed = time.time() - t0
        
        # ----------------- [수정된 부분] -----------------
        # 추천 결과에 image_url이 없거나 비어있는 경우, Spotify API로 채워 넣습니다.
        if "image_url" not in TopN_result.columns or TopN_result["image_url"].isnull().all():
             TopN_result = enrich_with_image_url(TopN_result)
        # ----------------- [수정 끝] -----------------
        
        # 여기서 image_url이 TopN_result에 이미 포함되어 있다고 가정합니다. (TopN.py 수정 예정)
        
        return {"items": TopN_result.to_dict(orient="records"), "elapsed_time": elapsed}
    except Exception as e:
        print(f"❌ Error in /recommend_ranked: {e}") 
        raise HTTPException(status_code=400, detail=f"Recommendation failed: {str(e)}")
