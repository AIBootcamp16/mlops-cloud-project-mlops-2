from __future__ import annotations

import os
from typing import Sequence
from dataclasses import dataclass

_IMPORT_ERR = None
try:
    from .mysql_logger import MySQLLogger
except Exception as e:
    MySQLLogger = None
    _IMPORT_ERR = e

@dataclass
class RecommendLog:
    by_field: str
    query: str
    top_k: int
    elapsed_sec: float
    seed_track_ids: Sequence[str]
    returned_track_ids: Sequence[str]

class RecoLogger:
    def __init__(self):
        self.enabled = bool(MySQLLogger)
        self.debug = os.getenv("RECO_LOG_DEBUG", "0") == "1"
        self.file_fallback_path = os.getenv("RECO_LOG_FALLBACK", "./logs/reco_logs.csv")

        if self.debug:
            print(f"[RecoLogger] enabled={self.enabled}")
            if not self.enabled and _IMPORT_ERR:
                print(f"[RecoLogger] mysql_logger import error: {_IMPORT_ERR}")

        if self.enabled:
            try:
                self.client = MySQLLogger(
                    host=os.getenv("MYSQL_HOST", "114.203.195.166"),
                    user=os.getenv("MYSQL_USER", "root"),
                    password=os.getenv("MYSQL_PASSWORD", "root"),
                    database=os.getenv("MYSQL_DB", "mlops"),
                    port=int(os.getenv("MYSQL_PORT", "3306")),
                )
                if self.debug:
                    print("[RecoLogger] MySQLLogger connected")
            except Exception as e:
                self.enabled = False
                if self.debug:
                    print(f"[RecoLogger] MySQL connect failed → fallback to file. err={e}")

        # 파일 백업 디렉토리 준비
        if not os.path.exists(os.path.dirname(self.file_fallback_path) or "."):
            os.makedirs(os.path.dirname(self.file_fallback_path), exist_ok=True)

    def log_recommend(self, log: RecommendLog):
        # 1) MySQL 시도
        if self.enabled:
            try:
                self.client.log_recommend(
                    by_field=log.by_field,
                    query=log.query,
                    top_k=log.top_k,
                    elapsed_sec=log.elapsed_sec,
                    seed_track_ids=list(log.seed_track_ids),
                    returned_track_ids=list(log.returned_track_ids),
                )
                if self.debug:
                    print("[RecoLogger] MySQL log OK")
                return
            except Exception as e:
                if self.debug:
                    print(f"[RecoLogger] MySQL log failed → fallback to file. err={e}")

        # 2) 파일 백업 (CSV)
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = [
            ts, log.by_field, log.query, log.top_k, log.elapsed_sec,
            "|".join(log.seed_track_ids), "|".join(log.returned_track_ids)
        ]
        header = ["ts","by_field","query","top_k","elapsed_sec","seed_track_ids","returned_track_ids"]
        write_header = not os.path.exists(self.file_fallback_path)

        with open(self.file_fallback_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
        if self.debug:
            print(f"[RecoLogger] File log OK → {self.file_fallback_path}")