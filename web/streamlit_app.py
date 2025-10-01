# ---------------------
# 실행 방법
# ---------------------
# # api 서버 (프로젝트 root에서 실행)
# python -m uvicorn src.api.api:app --reload --port 8000
# # UI (web 디렉토리에서 실행)
# python -m streamlit run .\streamlit_app.py

from __future__ import annotations

import os
from contextlib import contextmanager

import os
import json
import time
import requests
import pandas as pd
import streamlit as st
import pymysql

# ---------------------
# 로딩마스크
# ---------------------
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

# ---------------------
# 설정
# ---------------------
st.set_page_config(page_title="음악 추천", layout="wide")

DEFAULT_API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_BASE_DEFAULT = st.secrets.get("API_BASE", DEFAULT_API_BASE)

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
st.title("🎵 Tune for You 당신을 위한 음악 추천")

api_col = st.container()
with api_col:
    API_BASE = st.text_input(
        "API_BASE",
        value=API_BASE_DEFAULT,
        help="예: http://localhost:8000",
    ).strip()

tab1, tab2 = st.tabs(["🎶 음악 추천", "📊 품질 모니터링"])

# ---------------------
# Helpers
# ---------------------
def call_search(api_base: str, by: str, query: str, limit: int = 50) -> pd.DataFrame:
    """Backend /search 호출 (API에 /search 엔드포인트가 있어야 합니다)"""
    url = f"{api_base.rstrip('/')}/search"
    payload = {"by": by, "query": query, "limit": int(limit)}
    resp = requests.post(url, json=payload, timeout=2000)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return pd.DataFrame(items)

def call_recommend(api_base: str, by: str, query: str, top_k: int, seed_max: int | None = 1) -> pd.DataFrame:
    """Backend /recommend 호출"""
    url = f"{api_base.rstrip('/')}/recommend_ranked"
    payload = {"by": by, "query": query, "top_k": int(top_k)}
    if seed_max is not None:
        payload["seed_max"] = int(seed_max)  # 모델 패치 시 사용
    resp = requests.post(url, json=payload, timeout=2000)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return pd.DataFrame(items)

# ---------------------
# 탭 1: 음악 추천 화면 (API 호출)
# 1) 검색
# ---------------------
with tab1:
    # st.header("1) 음악 검색")

    query_by_map = {"노래 제목": "track_name", "가수 이름": "artist_name"}
    col1, col2 = st.columns([1, 3])
    with col1:
        query_by_label = st.selectbox("검색 기준", list(query_by_map.keys()), index=0)
        query_by = query_by_map[query_by_label]
    with col2:
        query_text = st.text_input("검색어", placeholder="노래 제목 또는 가수 이름을 입력하세요")

    # 표시 개수 → 내부 기본값 사용
    SEARCH_LIMIT = 50

    if st.button("검색"):
        if not query_text.strip():
            st.warning("검색어를 입력해 주세요.")
        else:
            try:
                with tiny_loading("검색 중..."):
                    t0 = time.time()
                    df_search = call_search(API_BASE, query_by, query_text, SEARCH_LIMIT)
                    elapsed = time.time() - t0
                if df_search.empty:
                    st.info("검색 결과가 없습니다. ({elapsed:.3f}초)")
                    st.session_state["search_df"] = None
                else:
                    st.session_state["search_df"] = df_search
                    st.success(f"총 {len(df_search)}건을 찾았습니다. ({elapsed:.3f}초)")
            except requests.exceptions.ConnectionError:
                st.error(f"API에 연결할 수 없습니다: {API_BASE}")
            except requests.exceptions.HTTPError as e:
                st.error(f"API 오류: {e.response.status_code} {e.response.text}")
            except Exception as e:
                st.exception(e)

    # 검색 결과 표시 & 표에서 행 선택
    df_search = st.session_state.get("search_df", None)
    selected_seed_id = None

    if isinstance(df_search, pd.DataFrame) and not df_search.empty:
        st.subheader("검색 결과")

        display_cols = [c for c in ["track_name", "artist_name"] if c in df_search.columns]
        if not display_cols:
            display_cols = [c for c in df_search.columns if c != "track_id"]

        # 원본 인덱스를 유지하여 선택 행에서 track_id를 역추적
        temp = df_search[display_cols].copy()
        temp.insert(0, "선택", False)

        edited = st.data_editor(
            temp,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=[c for c in temp.columns if c != "선택"],
            column_config={
                "선택": st.column_config.CheckboxColumn(
                    "선택",
                    help="추천을 생성할 곡을 한 곡만 선택하세요",
                    default=False,
                    required=False,
                )
            },
            key="search_table",
        )

        # 단일 선택 강제: 여러 개 체크되면 경고
        selected_rows = edited.index[edited["선택"] == True].tolist()
        if len(selected_rows) > 1:
            st.warning("한 곡만 선택해 주세요.")
        elif len(selected_rows) == 1:
            if "track_id" in df_search.columns:
                selected_seed_id = df_search.loc[selected_rows[0], "track_id"]
                st.session_state["selected_seed_id"] = selected_seed_id
            else:
                st.warning("백엔드 응답에 track_id가 없어 시드 선택을 완료할 수 없습니다.")

    # ---------------------
    # 2) 선택한 곡으로 추천 생성
    # ---------------------
    # st.header("2) 선택한 곡으로 추천 생성")

    # top_k = st.slider("상위 K개", min_value=1, max_value=100, value=10, step=1)
    top_k = 10
    if st.button("추천해줘!"):
        seed_id = st.session_state.get("selected_seed_id")
        if not seed_id:
            st.warning("먼저 검색 결과 표에서 곡을 하나 선택해 주세요.")
        else:
            try:
                with tiny_loading("추천 생성 중..."):
                    t0 = time.time()
                    df_rec = call_recommend(API_BASE, by="track_id", query=seed_id, top_k=top_k, seed_max=1)
                    elapsed = time.time() - t0
                if df_rec.empty:
                    st.info("추천 결과가 없습니다.")
                else:
                    st.success(f"총 {len(df_rec)}건의 추천을 생성했습니다. ({elapsed:.3f}초)")
                    prefer = [c for c in ["rank", "track_name", "artist_name", "distance"] if c in df_rec.columns]
                    other = [c for c in df_rec.columns if c not in prefer and c != "track_id"]
                    st.dataframe(df_rec[prefer + other], use_container_width=True, hide_index=True)
            except requests.exceptions.ConnectionError:
                st.error(f"API에 연결할 수 없습니다: {API_BASE}")
            except requests.exceptions.HTTPError as e:
                st.error(f"API 오류: {e.response.status_code} {e.response.text}")
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