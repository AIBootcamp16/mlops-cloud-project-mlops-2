# streamlit_app.py
import sys
sys.path.append("../src/model")

import streamlit as st
import sqlite3, pandas as pd, json, time
from spotify_recommender import Recommender

# ==============================
# 설정
# ==============================
DB_PATH = "./logs/reco_logs.sqlite3"
MODEL_DIR = "./models"

# ==============================
# 앱 레이아웃
# ==============================
st.set_page_config(page_title="Music Recommender & Dashboard", layout="wide")
st.title("🎵 Music Recommender System")

tab1, tab2 = st.tabs(["🎶 Music Recommendation", "📊 Quality Monitoring"])

# ==============================
# 탭 1: 음악 추천 화면
# ==============================
with tab1:
    st.header("Find Similar Songs")
    st.write("Type a track name, artist name, or track ID to get recommendations.")

    # 사용자 입력
    query_by = st.selectbox("Search by", ["track_name", "artist_name", "track_id"])
    query_txt = st.text_input("Enter query", "")
    top_k = st.slider("Top-K recommendations", 5, 20, 10)

    if st.button("Recommend"):
        if query_txt.strip():
            try:
                # 모델 로드
                rec = Recommender.load(MODEL_DIR)
                start = time.time()
                results = rec.recommend(by=query_by, query=query_txt, top_k=top_k)
                elapsed = time.time() - start

                st.success(f"Recommendations generated in {elapsed:.3f}s")
                st.dataframe(results)

            except Exception as e:
                st.error(f"Recommendation failed: {e}")
        else:
            st.warning("Please enter a query string.")

# ==============================
# 탭 2: 품질 모니터링 대시보드
# ==============================
with tab2:
    st.header("Recommendation Quality Dashboard")

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM recommend_logs ORDER BY id DESC LIMIT 2000", con)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])

    if df.empty:
        st.info("No logs found yet. Try making some recommendations first.")
    else:
        # Diversity 계산
        def calc_diversity(row):
            try:
                ids = json.loads(row["returned_track_ids"])
                return len(set(ids)) / max(len(ids), 1)
            except Exception:
                return None

        df["diversity"] = df.apply(calc_diversity, axis=1)

        # 요약 카드
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Diversity", f"{df['diversity'].mean():.3f}")
        col2.metric("Avg Latency (s)", f"{df['elapsed_sec'].mean():.3f}")
        col3.metric("Total Recs", len(df))

        # Diversity 트렌드
        st.subheader("Diversity Over Time")
        st.line_chart(df.set_index("ts_utc")["diversity"])

        # Latency 트렌드
        st.subheader("Latency Over Time")
        st.line_chart(df.set_index("ts_utc")["elapsed_sec"])

        # 최근 추천 로그 테이블
        st.subheader("Recent Recommendations")
        st.dataframe(df[["ts_utc","by_field","query","returned_track_ids","diversity"]].head(20))
