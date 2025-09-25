# streamlit_app.py
import sys
sys.path.append("../src/model")

import streamlit as st
import sqlite3
import pandas as pd
import json
import time
from spotify_recommender import Recommender
import pymysql

# ==============================
# 설정
# ==============================
MODEL_DIR = "../models"
DB_CFG = dict(
    host="114.203.195.166",
    user="root",
    password="root",
    database="mlops",
    port=3306,
    charset="utf8mb4",
)

# ==============================
# 앱 레이아웃
# ==============================
st.set_page_config(page_title="음악 추천 & 대시보드", layout="wide")
st.title("🎵 Tune for You 당신을 위한 음악 추천")

tab1, tab2 = st.tabs(["🎶 음악 추천", "📊 품질 모니터링"])

# ==============================
# 탭 1: 음악 추천 화면
# ==============================
with tab1:
    st.header("비슷한 음악을 찾아보세요!")
    st.write("듣고 싶은 노래 제목이나 가수 이름을 입력하면 추천해 드릴게요.")

    # 한국어 라벨 ↔ 내부 값 매핑
    query_by_options = {
        "노래 제목": "track_name",
        "가수 이름": "artist_name",
    }
    query_by_label = st.selectbox("검색 기준", list(query_by_options.keys()))
    query_by = query_by_options[query_by_label]

    query_txt = st.text_input("검색어", "", placeholder="노래 제목이나 가수 이름을 입력해 주세요")
    top_k = st.slider("상위 K개 추천 결과", min_value=5, max_value=20, value=10)

    if st.button("추천해줘!"):
        if query_txt.strip():
            try:
                # 모델 로드
                rec = Recommender.load(MODEL_DIR)
                start = time.time()
                results = rec.recommend(by=query_by, query=query_txt, top_k=top_k)
                elapsed = time.time() - start

                st.success(f"추천 결과가 {elapsed:.3f}초 만에 생성됐어요.")
                st.dataframe(results)
            except Exception as e:
                st.error(f"추천 생성에 실패했어요: {e}")
        else:
            st.warning("검색어를 입력해 주세요.")

# ==============================
# 탭 2: 품질 모니터링 대시보드
# ==============================
with tab2:
    st.header("추천 품질 대시보드")

    try:
        con = pymysql.connect(**DB_CFG)
        df = pd.read_sql("SELECT * FROM recommend_logs ORDER BY id DESC LIMIT 100", con)
    except Exception as e:
        st.error(f"로그 DB에 접속할 수 없습니다: {e}")
        df = pd.DataFrame()
    finally:
        try:
            con.close()
        except Exception:
            pass

    if df.empty:
        st.info("아직 로그가 없어요. 먼저 추천을 한 번 실행해 보세요.")
    else:
        # 타입 정리
        df["ts_utc"] = pd.to_datetime(df.get("ts_utc"), errors="coerce")
        df["elapsed_sec"] = pd.to_numeric(df.get("elapsed_sec"), errors="coerce")

        # 다양성(Diversity) 계산: returned_ids 가 JSON 문자열이라고 가정
        def calc_diversity(row):
            try:
                ids_raw = row.get("returned_ids")
                ids = json.loads(ids_raw) if isinstance(ids_raw, str) else ids_raw
                if not isinstance(ids, (list, tuple)):
                    return None
                return len(set(ids)) / max(len(ids), 1)
            except Exception:
                return None

        df["diversity"] = df.apply(calc_diversity, axis=1)

        st.dataframe(df)

        # 요약 카드
        col1, col2, col3 = st.columns(3)
        col1.metric("평균 다양성", f"{df['diversity'].mean():.3f}" if df["diversity"].notna().any() else "N/A")
        col2.metric("평균 지연시간(초)", f"{df['elapsed_sec'].mean():.3f}" if df["elapsed_sec"].notna().any() else "N/A")
        col3.metric("총 추천 수", len(df))

        # 다양성 추이
        st.subheader("다양성(Diversity) 추이")
        if df["ts_utc"].notna().any() and df["diversity"].notna().any():
            st.line_chart(df.set_index("ts_utc")["diversity"].dropna())
        else:
            st.caption("시각화할 데이터가 충분하지 않습니다.")

        # 지연시간 추이
        st.subheader("지연 시간(Latency) 추이")
        if df["ts_utc"].notna().any() and df["elapsed_sec"].notna().any():
            st.line_chart(df.set_index("ts_utc")["elapsed_sec"].dropna())
        else:
            st.caption("시각화할 데이터가 충분하지 않습니다.")

        # 최근 추천 로그 테이블
        st.subheader("최근 추천 로그")
        cols_to_show = [c for c in ["ts_utc", "by_field", "query", "returned_ids", "diversity"] if c in df.columns]
        st.dataframe(df[cols_to_show].head(20))
