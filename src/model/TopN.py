import os
import time
import pandas as pd

from pathlib import Path
from NModel import FAISSRecommender, LGBMRecommender

# file_path = os.path.abspath(__file__)  # /home/user/project/main.py
# dir_path = os.path.dirname(file_path)  # /home/user/project

#BASE_DIR = Path(__file__).resolve().parents[2]   # src/api/TopN.py 기준 두 단계 위
#DATA_PATH = BASE_DIR / "dataset" / "processed" / "spotify_data_clean.csv"

#df = pd.read_csv(DATA_PATH)

# 2. 사용할 피처 선택 (벡터화)
features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

#_df_indexed = df.set_index('track_id')

class TopN_Model:
    def __init__(self):
        self.FAISS_REC = FAISSRecommender.MLFLOWload()
        self.LGBM_REC = LGBMRecommender.MLFLOW_load()

    def Search(self, by: str, query: str, top_k: int = 5):
        recommend_item = self.FAISS_REC.recommend(by=by,query=query,top_k=100)
        # 2. DataFrame 필터링
        #df_filtered = _df_indexed.loc[recommend_item['track_id']]
        #X = recommend_item[features]
        #y = df_filtered['popularity']  # 인기도 컬럼
        y_pred = self.LGBM_REC.predict(recommend_item)
        recommend_item['popularity'] = y_pred 
        #recommend_item.drop(features)
        recommend_item.drop(columns=features, inplace=True)
        TopN = recommend_item.nlargest(top_k, 'popularity')
        return TopN


if __name__ == "__main__":
    myTopN_Model = TopN_Model()
    starttime = time.time()
    TopN_result = myTopN_Model.Search(by="track_name", query="Shape of You", top_k=5)
    print(time.time() - starttime)
    print(TopN_result)