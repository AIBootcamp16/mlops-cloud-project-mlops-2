CREATE TABLE IF NOT EXISTS recommend_logs (
  id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts_utc              DATETIME NOT NULL,       -- 추천 시각(UTC)
  by_field            VARCHAR(50),             -- track_id | track_name | artist_name
  query               VARCHAR(255),            -- 검색어/시드표현
  top_k               INT,
  elapsed_sec         DOUBLE,                  -- 처리 시간(초)
  seed_track_ids      JSON,                    -- ["id1","id2",...]
  returned_track_ids  JSON,                    -- ["idA","idB",...]
  diversity           DOUBLE,                  -- len(unique)/len(all)
  avg_popularity      DOUBLE,                  -- 추천 인기도 평균
  genre_precision     DOUBLE,                  -- 시드장르 일치율
  KEY idx_ts (ts_utc),
  KEY idx_by (by_field)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 추천 아이템 테이블 (rank에 백틱 적용)
CREATE TABLE IF NOT EXISTS recommend_items (
  id       BIGINT AUTO_INCREMENT PRIMARY KEY,
  log_id   BIGINT NOT NULL,                -- FK to recommend_logs.id
  `rank`   INT NOT NULL,                   -- 1..top_k
  track_id VARCHAR(64) NOT NULL,           -- spotify_music.track_id
  distance DOUBLE NULL,                    -- (옵션) 모델 거리

  CONSTRAINT fk_reco_items_log
    FOREIGN KEY (log_id) REFERENCES recommend_logs(id) ON DELETE CASCADE,
  KEY idx_log (log_id),
  KEY idx_tid (track_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



-- SHOW GRANTS FOR 'root'@'%';


