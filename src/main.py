from __future__ import annotations
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 1. 경로 설정 및 임포트 경로 수정 (최상단으로 이동)
# ----------------------------------------------------
# BASE_DIR을 프로젝트 루트로 정의합니다.
BASE_DIR = Path(__file__).resolve().parent.parent 
SRC_DIR = Path(__file__).resolve().parent # src 디렉토리

# 🚨 긴급 조치: Uvicorn의 --reload 서브프로세스에서 로컬 모듈을 올바르게 찾도록
# 'src 디렉토리'를 sys.path의 맨 앞에 다시 추가합니다. (가장 안정적인 우회책)
SRC_DIR_STR = str(SRC_DIR)
if SRC_DIR_STR not in sys.path:
    sys.path.insert(0, SRC_DIR_STR)

print("--- STARTUP DEBUG INFO ---")
print(f"Project Root: {BASE_DIR}")
print(f"SRC Directory (re-added to path): {SRC_DIR_STR}")
print("--- END DEBUG INFO ---")


# ----------------------------------------------------
# 2. 로컬 모듈 임포트 (단순 모듈 이름 사용)
# ----------------------------------------------------
# ----------------------------------------------------
# 3. 환경 변수 로드 (Spotify 인증 문제 해결)
# ----------------------------------------------------
# 프로젝트 루트에 있는 .env 파일을 로드하여 환경 변수(SPOTIPY_CLIENT_ID 등)를 설정합니다.
load_dotenv(BASE_DIR / ".env") 

# ----------------------------------------------------
# 4. 모델 및 데이터 경로 설정 
# ----------------------------------------------------
MODEL_DIR = BASE_DIR / "models" # 모델 아티팩트 경로
DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv" # 데이터셋 경로


# ----------------------------------------------------
# 기존 Import 재정렬
# ----------------------------------------------------
import pandas as pd
from pydantic import BaseModel, Field

# 🚨 수정 2: 'src' 디렉토리가 sys.path에 있으므로, 'from src.model.NModel' 대신
# 단순 서브패키지 경로로 임포트합니다.
from model.NModel import FAISSRecommender, LGBMRecommender, Finder, TopN_Model 

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests # Spotify API 호출에 사용

# -------------------- [전역 상태 및 경로 설정] --------------------

# 전역 상태 관리 클래스
class GlobalState:
    def __init__(self):
        # 모델 인스턴스
        self.finder: Optional[Finder] = None
        self.faiss_rec: Optional[FAISSRecommender] = None
        self.lgbm_rec: Optional[LGBMRecommender] = None
        self.topn_model: Optional[TopN_Model] = None
        self.df: Optional[pd.DataFrame] = None
        # 인증 상태
        self.sp: Optional[spotipy.Spotify] = None
        self.spotipy_error: Optional[str] = None
        # 로드 성공 여부 플래그
        self.is_model_loaded: bool = False

g = GlobalState()

# -------------------- [API 요청 스키마] --------------------

class SearchRequest(BaseModel):
    # Streamlit에서 by, query, limit을 모두 보냅니다.
    by: str = Field("track_name", description="검색 기준: track_name, artist_name, track_id 중 하나")
    query: str = Field(..., description="검색할 트랙 이름, 아티스트 이름 또는 트랙 ID")
    limit: int = Field(50, description="반환할 검색 결과 개수", ge=1, le=50)

class RecommendRankedRequest(BaseModel):
    by: str = Field(..., description="검색 기준 (일반적으로 track_id)")
    query: str = Field(..., description="검색어 (일반적으로 시드 트랙 ID)")
    top_k: int = Field(10, description="반환할 추천 결과 개수", ge=1, le=50)

# -------------------- [Spotify 이미지 URL 보강 함수] --------------------

def enrich_with_image_url(df: pd.DataFrame) -> pd.DataFrame:
    """
    track_id를 사용하여 Spotify에서 image_url을 가져와 DataFrame에 추가합니다.
    (src/api.py의 로직을 그대로 가져옴)
    """
    if g.sp is None or g.spotipy_error:
        print(f"⚠️ Spotify API 비활성: {g.spotipy_error}")
        df["image_url"] = None
        return df

    if df.empty or "track_id" not in df.columns:
        df["image_url"] = None
        return df

    track_ids = df["track_id"].tolist()
    records = [None] * len(track_ids)
    
    # Spotify API는 한 번에 최대 50개의 track_id만 처리할 수 있습니다.
    for i in range(0, len(track_ids), 50):
        chunk_ids = track_ids[i:i+50]
        try:
            track_details = g.sp.tracks(chunk_ids)
            if track_details and track_details.get('tracks'):
                for j, t in enumerate(track_details['tracks']):
                    if t and t.get('album') and t['album'].get('images'):
                        # 가장 큰 이미지 URL (0번째)을 저장
                        records[i + j] = t['album']['images'][0]['url']
        except requests.exceptions.HTTPError as e:
            print(f"❌ Spotify API HTTP 오류: {e.response.status_code}")
        except Exception as e:
            print(f"❌ Spotify API 호출 중 오류 발생: {e}")

    df["image_url"] = records
    return df

# -------------------- [FastAPI Lifespan (시작/종료)] --------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- 🚀 서버 시작: 모델 및 아티팩트 로드 중 ---")
    
    # 1. Spotify 인증 설정 (src/api.py 로직)
    SPOTIPY_CLIENT_ID_VAL = os.environ.get('SPOTIPY_CLIENT_ID')
    SPOTIPY_CLIENT_SECRET_VAL = os.environ.get('SPOTIPY_CLIENT_SECRET')
    try:
        # 🚨 수정: SpotifyClientClientCredentials -> SpotifyClientCredentials (오타 수정)
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID_VAL,
            client_secret=SPOTIPY_CLIENT_SECRET_VAL
        )
        g.sp = spotipy.Spotify(auth_manager=auth_manager)
        g.spotipy_error = None
        print("✅ Spotify 인증 성공.")
    except Exception as e:
        g.sp = None
        g.spotipy_error = str(e)
        print(f"❌ Spotify 인증 실패. 오류 메시지: {e}")

    # 2. 데이터 로드 (src/api.py 로직)
    try:
        if not DATA_PATH.exists():
            raise RuntimeError(f"DATA not found: {DATA_PATH}")
        g.df = pd.read_csv(DATA_PATH)
        if "track_id" not in g.df.columns:
            raise RuntimeError("dataset must contain 'track_id' column")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during Data load: {e}")
        raise RuntimeError(f"데이터 로드 실패로 인해 서버를 시작할 수 없습니다: {e}")

    # 3. 모델 아티팩트 로드 및 초기화 (src/api.py 로직)
    try:
        # ----------------------------------------------------------------------
        # DEBUGGING: 현재 모델 디렉토리에 존재하는 파일 목록을 출력합니다.
        model_files = [f.name for f in MODEL_DIR.iterdir() if f.is_file()]
        print(f"DEBUG: '{MODEL_DIR.name}' 디렉토리 파일 목록: {model_files}")
        # ----------------------------------------------------------------------

        g.finder = Finder.load(str(MODEL_DIR))
        g.faiss_rec = FAISSRecommender.load(str(MODEL_DIR))
        g.lgbm_rec = LGBMRecommender.load(str(MODEL_DIR))
        
        # TopN 모델 초기화
        g.topn_model = TopN_Model(
            finder=g.finder, 
            faiss_recommender=g.faiss_rec, 
            lgbm_recommender=g.lgbm_rec, 
            data_df=g.df
        )

        g.is_model_loaded = True
        print("--- ✅ 모델 로드 성공: 모든 모델 및 아티팩트 로드 완료 ---")
        
    except Exception as e:
        # 모델 로드 또는 초기화 실패 시
        print(f"--- ❌ 치명적인 모델 로드 실패: {e} ---")
        g.is_model_loaded = False
        raise RuntimeError(f"모델 로드 실패로 인해 서버를 시작할 수 없습니다: {e}") 

    yield
    # 서버 종료 시 필요한 정리 작업이 있다면 여기에 추가

# -------------------- [FastAPI 앱 생성] --------------------

app = FastAPI(
    title="Spotipy Music Recommender API",
    description="음악 트랙을 기반으로 유사한 트랙을 추천합니다 (FAISS + LGBM 기반).",
    version="1.0.0",
    lifespan=lifespan # 라이프사이클 핸들러 연결
)

# -------------------- [API 엔드포인트 정의] --------------------

# 헬스 체크 엔드포인트
@app.get("/health")
def health_check():
    """서버 상태 및 모델 로드 상태를 확인합니다."""
    if not g.is_model_loaded:
        return {"status": "degraded", "message": "Recommender model failed to load during startup."}
    return {"status": "ok", "message": "Recommender model loaded successfully."}

# 음악 검색 엔드포인트 (/search)
@app.post("/search")
def search(req: SearchRequest):
    """주어진 쿼리를 기반으로 트랙을 검색합니다. (Streamlit의 1단계)"""
    
    if g.finder is None:
        raise HTTPException(status_code=500, detail="Finder is not initialized (missing model artifacts).")
        
    try:
        # Finder를 사용하여 검색
        matches = g.finder._lookup_indices(by=req.by, query=req.query, max_matches=req.limit)
        idx = g.finder.artifacts.id_index # 모든 메타데이터가 담긴 데이터프레임
        
        # Streamlit에서 요구하는 컬럼만 포함
        cols = [c for c in ["track_id", "track_name", "artist_name", "image_url"] if c in idx.columns]
        df = idx.iloc[matches][cols].reset_index(drop=True)

        # 데이터 파일에 image_url이 없거나 비어있는 경우, Spotify API로 채워 넣습니다.
        df = enrich_with_image_url(df)

        # Streamlit의 call_search가 기대하는 {"items": [...]} 형식으로 반환
        return {"items": df.to_dict(orient="records")}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"'{req.query}'에 대한 결과를 찾을 수 없습니다: {str(e)}")
    except Exception as e:
        print(f"FATAL ERROR in /search: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 예상치 못한 오류 발생: {str(e)}")


# 음악 추천 및 재순위 엔드포인트 (/recommend_ranked)
@app.post("/recommend_ranked")
def recommend_ranked(req: RecommendRankedRequest):
    """선택된 곡을 기반으로 추천을 생성하고 재순위합니다. (Streamlit의 2단계)"""
    
    if g.topn_model is None:
        raise HTTPException(status_code=500, detail="TopN Model is not initialized (missing model artifacts).")
        
    try:
        t0 = time.time()
        
        # TopN_Model의 Search 함수를 사용하여 추천 및 재순위 실행
        TopN_result = g.topn_model.Search(
            by=req.by, 
            query=req.query, 
            top_k=req.top_k
        )
        elapsed = time.time() - t0
        
        # 추천 결과에 image_url이 없거나 비어있는 경우, Spotify API로 채워 넣습니다.
        TopN_result = enrich_with_image_url(TopN_result)
        
        # Streamlit의 call_recommend가 기대하는 {"items": [...]} 형식으로 반환
        return {"items": TopN_result.to_dict(orient="records"), "elapsed_time": elapsed}
        
    except Exception as e:
        print(f"❌ Error in /recommend_ranked: {e}") 
        raise HTTPException(status_code=400, detail=f"추천 실패: {str(e)}")


@app.get("/")
async def root_status():
    """서버 상태 및 모델 로드 정보를 반환합니다."""
    is_ready = g.is_model_loaded
    
    status = "ok" if is_ready else "loading_or_error"
    message = "Spotify Recommender API is running successfully." if is_ready else "API is starting, models may not be ready yet."
    
    return {
        "status": status, 
        "message": message,
        "api_version": "1.0",
        "models_ready": is_ready
    }
