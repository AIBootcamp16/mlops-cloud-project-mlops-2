from __future__ import annotations

import os
import json
import time
import requests
import pandas as pd
import streamlit as st
import pymysql

from contextlib import contextmanager
import streamlit as st

# ==============================
# 로딩마스크
# ==============================
@contextmanager
def tiny_loading(text: str = "로딩 중..."):
    ph = st.empty()
    ph.markdown(
        f"""
        <style>
        .tiny-loader {{
            display:inline-flex; align-items:center; gap:8px;
            padding:6px 10px; border-radius:9999px;
            background:rgba(0,0,0,0.08); font-size:13px;
        }}
        .tiny-spinner {{
            width:14px; height:14px; border-radius:50%;
            border:2px solid rgba(0,0,0,0.15);
            border-top-color: rgba(0,0,0,0.6);
            animation: tiny-rot 0.8s linear infinite;
        }}
        @keyframes tiny-rot {{ to {{ transform: rotate(360deg); }} }}
        </style>
        <span class="tiny-loader">
          <span class="tiny-spinner"></span>{text}
        </span>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        ph.empty()

# ==============================
# 설정
# ==============================
# ➜ API 주소: secrets.toml 또는 환경변수(API_BASE) → 기본값 순
DEFAULT_API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_BASE = st.secrets.get("API_BASE", DEFAULT_API_BASE)

# DB 설정
DB_CFG = dict(
    host=os.environ.get("MYSQL_HOST", "114.203.195.166"),
    user=os.environ.get("MYSQL_USER", "root"),
    password=os.environ.get("MYSQL_PASSWORD", "root"),
    database=os.environ.get("MYSQL_DB", "mlops"),
    port=int(os.environ.get("MYSQL_PORT", "3306")),
    charset="utf8mb4",
)

# (선택) 파일 폴백 로그 경로
FILE_FALLBACK_LOG = os.environ.get("RECO_LOG_FALLBACK", "./logs/reco_logs.csv")

# ==============================
# 앱 레이아웃
# ==============================
st.set_page_config(page_title="음악 추천 & 대시보드", layout="wide")
st.title("🎵 Tune for You 당신을 위한 음악 추천")

with st.sidebar:
    st.subheader("Settings")
    API_BASE_input = st.text_input("API_BASE", value=API_BASE, help="예: http://localhost:8000")
    if API_BASE_input:
        API_BASE = API_BASE_input

tab1, tab2 = st.tabs(["🎶 음악 추천", "📊 품질 모니터링"])

# ==============================
# 탭 1: 음악 추천 화면 (API 호출)
# ==============================
with tab1:
    st.header("비슷한 음악을 찾아보세요!")
    st.write("듣고 싶은 노래 제목이나 가수 이름을 입력하면 추천해 드릴게요.")

    query_by_options = {
        "노래 제목": "track_name",
        "가수 이름": "artist_name",
        "트랙 ID": "track_id",
    }
    query_by_label = st.selectbox("검색 기준", list(query_by_options.keys()))
    query_by = query_by_options[query_by_label]

    query_txt = st.text_input("검색어", "", placeholder="노래 제목·가수·트랙ID를 입력해 주세요")
    top_k = st.slider("상위 K개 추천 결과", min_value=5, max_value=50, value=10)

    if st.button("추천해줘!"):
        if not query_txt.strip():
            st.warning("검색어를 입력해 주세요.")
        else:
            try:
                with tiny_loading("추천 생성 중..."):
                    t0 = time.time()
                    resp = requests.post(
                        f"{API_BASE.rstrip('/')}/recommend",
                        json={"by": query_by, "query": query_txt, "top_k": int(top_k)},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    items = resp.json().get("items", [])
                    elapsed = time.time() - t0

                    if not items:
                        st.info("추천 결과가 없습니다.")
                    else:
                        df = pd.DataFrame(items)
                        st.success(f"추천 결과가 {elapsed:.3f}초 만에 생성됐어요.")
                        # 주요 칼럼 우선 정렬
                        prefer = [c for c in ["rank", "track_id", "track_name", "artist_name", "distance"] if c in df.columns]
                        other = [c for c in df.columns if c not in prefer]
                        st.dataframe(df[prefer + other], use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error(f"API에 연결할 수 없습니다: {API_BASE}")
                st.code(
                    "python -m uvicorn src.api.api:app --reload --port 8000",
                    language="bash",
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API 오류: {e.response.status_code} {e.response.text}")
            except requests.exceptions.Timeout:
                st.error("요청이 시간 초과되었습니다. 잠시 후 다시 시도해 주세요.")
            except Exception as e:
                st.exception(e)

# ==============================
# 탭 2: 품질 모니터링 (MySQL → 파일 폴백)
# ==============================
with tab2:
    st.header("추천 품질 대시보드")

    df = pd.DataFrame()
    db_err = None
    try:
        con = pymysql.connect(**DB_CFG)
        df = pd.read_sql("SELECT * FROM recommend_logs ORDER BY id DESC LIMIT 200", con)
    except Exception as e:
        db_err = e
    finally:
        try:
            con.close()
        except Exception:
            pass

    if df.empty and db_err is not None:
        st.warning(f"DB에서 로그를 불러오지 못했습니다: {db_err}")
        # 파일 폴백 시도
        if os.path.exists(FILE_FALLBACK_LOG):
            try:
                df = pd.read_csv(FILE_FALLBACK_LOG)
                st.info(f"파일 폴백 로그를 사용합니다: {FILE_FALLBACK_LOG}")
            except Exception as e:
                st.error(f"폴백 로그도 읽을 수 없습니다: {e}")

    if df.empty:
        st.info("아직 로그가 없어요. 먼저 추천을 한 번 실행해 보세요.")
    else:
        # 스키마 정리: ts(datetime), elapsed_sec(float), seed/returned(JSON 또는 문자열)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        elif "ts_utc" in df.columns:
            df["ts"] = pd.to_datetime(df["ts_utc"], errors="coerce")
        else:
            # 둘 다 없으면 인덱스 기준 가짜시간
            df["ts"] = pd.to_datetime(pd.Timestamp.now())

        df["elapsed_sec"] = pd.to_numeric(df.get("elapsed_sec"), errors="coerce")

        def parse_json_maybe(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except Exception:
                    # 파이프(|)로 합친 파일 폴백 포맷일 수 있음
                    if "|" in x:
                        return [t for t in x.split("|") if t]
            return []

        df["seed_track_ids"] = df.get("seed_track_ids", []).apply(parse_json_maybe) if "seed_track_ids" in df.columns else [[]]*len(df)
        df["returned_track_ids"] = df.get("returned_track_ids", []).apply(parse_json_maybe) if "returned_track_ids" in df.columns else [[]]*len(df)

        # 다양성 지표: 고유 추천 수 / 전체 추천 수
        def calc_diversity(lst):
            try:
                return len(set(lst)) / max(len(lst), 1)
            except Exception:
                return None

        df["diversity"] = df["returned_track_ids"].apply(calc_diversity)

        # 보여주기
        st.subheader("요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("평균 다양성", f"{pd.to_numeric(df['diversity'], errors='coerce').mean():.3f}")
        c2.metric("평균 지연시간(초)", f"{pd.to_numeric(df['elapsed_sec'], errors='coerce').mean():.3f}")
        c3.metric("총 추천 수", len(df))

        st.subheader("다양성 추이")
        div_series = df.set_index("ts")["diversity"].dropna()
        if not div_series.empty:
            st.line_chart(div_series)
        else:
            st.caption("시각화할 데이터가 충분하지 않습니다.")

        st.subheader("지연시간 추이")
        lat_series = df.set_index("ts")["elapsed_sec"].dropna()
        if not lat_series.empty:
            st.line_chart(lat_series)
        else:
            st.caption("시각화할 데이터가 충분하지 않습니다.")

        st.subheader("최근 추천 로그")
        show_cols = [c for c in ["ts", "by_field", "query", "top_k", "elapsed_sec"] if c in df.columns]
        st.dataframe(df[show_cols].head(30), use_container_width=True)

# st.caption(f"API base: {API_BASE}")
# st.caption("CORS 에러가 나면 FastAPI에 CORSMiddleware를 추가하세요 (허용 origin에 http://localhost:8501).")
