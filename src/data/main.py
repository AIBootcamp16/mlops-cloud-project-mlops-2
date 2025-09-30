from preprocessing import MusicDataPreProcessor

def main():
    pp = MusicDataPreProcessor(id_col="track_id")

    file_path = "../../dataset/raw/spotify_data.csv"

    # 1. 로드
    pp.load_data(file_path)

    # 2. 전처리
    clean = pp.preprocess(dropna=True, dedupe=True)

    # 3. 저장
    pp.save("../../dataset/processed/spotify_data_clean.csv")

    print("전처리 완료:", clean.shape)


if __name__ == "__main__":
    main()
