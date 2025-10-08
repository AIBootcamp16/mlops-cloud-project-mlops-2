from __future__ import annotations
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ----------------------------------------------------
# 1️⃣ 경로 설정 및 환경 변수 로드
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv"

# ----------------------------------------------------
# 2️⃣ 모델 모듈 임포트
# ----------------------------------------------------
from model.NModel import FAISSRecommender, LGBMRecommender, Finder, TopN_Model


# ----------------------------------------------------
# 3️⃣ 전역 상태 관리 클래스
# ----------------------------------------------------
class GlobalState:
    def __init__(self):
        self.finder: Optional[Finder] = None
        self.faiss_rec: Optional[FAISSRecommender] = None
        self.lgbm_rec: Optional[LGBMRecommender] = None
        self.topn_model: Optional[TopN_Model] = None
        self.df: Optional[pd.DataFrame] = None
        self.sp: Optional[spotipy.Spotify] = None
        self.spotipy_error: Optional[str] = None
        self.is_model_loaded: bool = False


g = GlobalState()

# ----------------------------------------------------
# 4️⃣ Pydantic 요청 모델 정의
# ----------------------------------------------------
class SearchRequest(BaseModel):
    by: str = Field("track_name", description="검색 기준 (track_name, artist_name, track_id)")
    query: str = Field(..., description="검색어")
    limit: int = Field(50, ge=1, le=50, description="검색 결과 개수")


class RecommendRankedRequest(BaseModel):
    by: str = Field(..., description="검색 기준 (보통 track_id)")
    query: str = Field(..., description="시드 곡 ID")
    top_k: int = Field(10, ge=1, le=50, description="추천 개수")


# ----------------------------------------------------
# 5️⃣ Spotify 이미지 URL 보강 함수
# ----------------------------------------------------
def enrich_with_image_url(df: pd.DataFrame) -> pd.DataFrame:
    """Spotify API로부터 트랙 이미지 URL을 가져옵니다."""
    if g.sp is None or g.spotipy_error:
        df["image_url"] = None
        return df

    if df.empty or "track_id" not in df.columns:
        df["image_url"] = None
        return df

    records = [None] * len(df)
    track_ids = df["track_id"].tolist()
    for i in range(0, len(track_ids), 50):
        chunk_ids = track_ids[i:i + 50]
        try:
            details = g.sp.tracks(chunk_ids)
            if details and details.get("tracks"):
                for j, t in enumerate(details["tracks"]):
                    if t and t.get("album") and t["album"].get("images"):
                        records[i + j] = t["album"]["images"][0]["url"]
        except Exception:
            pass
    df["image_url"] = records
    return df


# ----------------------------------------------------
# 6️⃣ FastAPI lifespan (시작/종료 시점)
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- 🚀 서버 시작: 모델 및 아티팩트 로드 중 ---")

    # 1️⃣ Spotify 인증
    try:
        auth = SpotifyClientCredentials(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
        )
        g.sp = spotipy.Spotify(auth_manager=auth)
        print("✅ Spotify 인증 성공.")
    except Exception as e:
        g.sp = None
        g.spotipy_error = str(e)
        print(f"❌ Spotify 인증 실패: {e}")

    # 2️⃣ 데이터 로드 (없어도 서버는 종료되지 않음)
    try:
        if not DATA_PATH.exists():
            print(f"⚠️ DATASET WARNING: {DATA_PATH.name} 파일이 없습니다.")
            print("   👉 dataset/processed/ 디렉토리에 spotify_data_clean.csv를 추가하세요.")
            g.df = pd.DataFrame(columns=[
                "track_id", "track_name", "artist_name", "popularity"
            ])
        else:
            g.df = pd.read_csv(DATA_PATH)
            print(f"✅ 데이터 로드 성공: {len(g.df)}행")

        if "track_id" not in g.df.columns:
            print("⚠️ WARNING: 'track_id' 컬럼이 없습니다. 검색 기능이 제한될 수 있습니다.")
    except Exception as e:
        print(f"❌ 데이터 로드 오류 (무시됨): {e}")
        g.df = pd.DataFrame(columns=["track_id", "track_name", "artist_name", "popularity"])

    # 3️⃣ 모델 로드
    try:
        g.finder = Finder.load(str(MODEL_DIR))
        g.faiss_rec = FAISSRecommender.load(str(MODEL_DIR))
        g.lgbm_rec = LGBMRecommender.load(str(MODEL_DIR))
        g.topn_model = TopN_Model(
            finder=g.finder,
            faiss_recommender=g.faiss_rec,
            lgbm_recommender=g.lgbm_rec,
            data_df=g.df
        )
        g.is_model_loaded = True
        print("✅ 모델 로드 완료")
    except Exception as e:
        g.is_model_loaded = False
        print(f"❌ 모델 로드 실패 (서버는 계속 실행): {e}")

    yield
    print("🛑 서버 종료")


# ----------------------------------------------------
# 7️⃣ FastAPI 인스턴스 생성
# ----------------------------------------------------
app = FastAPI(
    title="Spotipy Music Recommender API",
    description="음악 트랙 기반 추천 시스템 (FAISS + LGBM)",
    version="2.0.0",
    lifespan=lifespan
)


# ----------------------------------------------------
# 8️⃣ API 엔드포인트
# ----------------------------------------------------
@app.get("/health")
def health_check():
    if not g.is_model_loaded:
        return {"status": "warning", "message": "Model not fully loaded."}
    return {"status": "ok", "message": "Service healthy."}


@app.post("/search")
def search(req: SearchRequest):
    if g.finder is None:
        raise HTTPException(status_code=500, detail="Finder not initialized.")
    try:
        matches = g.finder._lookup_indices(by=req.by, query=req.query, max_matches=req.limit)
        idx = g.finder.artifacts.id_index
        cols = [c for c in ["track_id", "track_name", "artist_name", "image_url"] if c in idx.columns]
        df = idx.iloc[matches][cols].reset_index(drop=True)
        df = enrich_with_image_url(df)
        return {"items": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")


@app.post("/recommend_ranked")
def recommend_ranked(req: RecommendRankedRequest):
    if g.topn_model is None:
        raise HTTPException(status_code=500, detail="TopN model not initialized.")
    try:
        t0 = time.time()
        res = g.topn_model.Search(by=req.by, query=req.query, top_k=req.top_k)
        elapsed = time.time() - t0
        res = enrich_with_image_url(res)
        return {"items": res.to_dict(orient="records"), "elapsed_time": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")


@app.get("/")
def root():
    return {
        "status": "ok" if g.is_model_loaded else "loading",
        "message": "Spotify Music Recommender API Ready.",
        "models_ready": g.is_model_loaded
    }
