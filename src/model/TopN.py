import os
import pandas as pd
from pathlib import Path
# Finder를 포함하여 필요한 모델 클래스를 NModel에서 임포트합니다.
from .NModel import FAISSRecommender, LGBMRecommender, Finder

# file_path = os.path.abspath(__file__)  # /home/user/project/main.py
# dir_path = os.path.dirname(file_path)  # /home/user/project

# U+00A0 특수 공백을 제거했습니다. (src/model/TopN.py 기준 두 단계 위 == /app/mlops)
BASE_DIR = Path(__file__).resolve().parents[2]    # MLOps 프로젝트 루트 경로
DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv"

# 주의: API 서버에서 data_df를 전달하므로 이 전역 로딩은 주석 처리합니다.
# df = pd.read_csv(DATA_PATH)

# 2. 사용할 피처 선택 (벡터화) - 클래스 내부에서 self.features로 사용
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]


class TopN_Model:
    # API에서 전달하는 모든 의존성 객체를 받도록 생성자 수정
    def __init__(self, finder: Finder, faiss_recommender: FAISSRecommender, lgbm_recommender: LGBMRecommender, data_df: pd.DataFrame):
        # 기존 MLFLOWload() 대신 주입받은 객체를 사용
        self.finder = finder
        self.FAISS_REC = faiss_recommender
        self.LGBM_REC = lgbm_recommender
        self.df = data_df # API에서 로드된 데이터프레임
        self.features = features # 피처 리스트

    # Search 메서드 내부에서 전역 변수(df, features) 대신 인스턴스 변수(self.df, self.features) 사용
    def Search(self, by: str, query: str, top_k: int = 5):
        # 1. FAISS를 이용해 후보 아이템 100개 검색
        recommend_item = self.FAISS_REC.recommend(by=by,query=query,top_k=100)
        
        # 2. 후보 아이템의 track_id를 이용해 인스턴스 데이터프레임(self.df)에서 데이터 추출
        # image_url을 포함한 모든 메타데이터를 가져옵니다.
        metadata_cols = [c for c in self.df.columns if c not in self.features]
        df_filtered = self.df.set_index('track_id').loc[recommend_item['track_id'], metadata_cols].reset_index()
        
        # 3. LGBM 모델의 입력 피처(self.features) 추출
        X = self.df.set_index('track_id').loc[recommend_item['track_id'], self.features]
        
        # 4. LGBM 모델로 재순위 점수 예측
        y_pred = self.LGBM_REC.predict(X)
        
        # 5. 예측 점수와 메타데이터를 검색 결과에 합치고, 점수 기준으로 TopN 선정
        # 추천 아이템(recommend_item)과 메타데이터(df_filtered)를 합칩니다.
        recommend_item['y_pred'] = y_pred 
        
        # track_id를 기준으로 병합하여 image_url을 가져옵니다.
        TopN = pd.merge(recommend_item, df_filtered, on='track_id', how='left')
        
        # TopN 선정
        TopN = TopN.nlargest(top_k, 'y_pred')
        
        # 최종 결과에 rank 컬럼 추가
        TopN['rank'] = range(1, len(TopN) + 1)
        
        return TopN
