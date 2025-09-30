import os

import pandas as pd

from model.NModel import FAISSRecommender
from model.NModel import LGBMRecommender


file_path = os.path.abspath(__file__)  # /home/user/project/main.py
dir_path = os.path.dirname(file_path)  # /home/user/project

df = pd.read_csv(f"{dir_path}/../dataset/processed/spotify_data_clean.csv")

# 2. 사용할 피처 선택 (벡터화)
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]


class TopN_Model:
    def __init__(self):
        self.FAISS_REC = FAISSRecommender.MLFLOWload()
        self.LGBM_REC = LGBMRecommender.MLFLOW_load()

    def Serch(self, by: str, query: str, top_k: int = 5):
        recommend_item = self.FAISS_REC.recommend(by=by,query=query,top_k=100)
        df_filtered = df.set_index('track_id').loc[recommend_item['track_id']].reset_index()
        X = df_filtered[features]
        #y = df_filtered['popularity']  # 인기도 컬럼
        y_pred = self.LGBM_REC.predict(X)
        recommend_item['y_pred'] = y_pred 
        TopN = recommend_item.nlargest(top_k, 'y_pred')
        return TopN


if __name__ == "__main__":
    myTopN_Model = TopN_Model()
    TopN_result = myTopN_Model.Serch(by="track_name", query="Shape of You", top_k=10)
    print(TopN_result)