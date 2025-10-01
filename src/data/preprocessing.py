import pandas as pd

class MusicDataPreProcessor:
    def __init__(self, id_col: str = "track_id"):
        self._data = pd.DataFrame()
        self._processed = pd.DataFrame()
        self.id_col = id_col

    def load_data(self, file_path: str) -> pd.DataFrame:
        self._data = pd.read_csv(file_path)
        return self._data

    def preprocess(self, dropna: bool = True, dedupe: bool = True) -> pd.DataFrame:
        if self._data.empty:
            raise ValueError("Call load_data() first.")
        df = self._data.copy()
        if dropna:
            df = df.dropna()
        if dedupe and self.id_col in df.columns:
            df = df.drop_duplicates(subset=[self.id_col])
        self._processed = df
        return df

    def save(self, file_path: str, index: bool = False) -> None:
        if self._processed.empty:
            raise ValueError("No processed data. Call preprocess() first.")
        self._processed.to_csv(file_path, index=index)

    def get_processed(self) -> pd.DataFrame:
        return self._processed.copy()
