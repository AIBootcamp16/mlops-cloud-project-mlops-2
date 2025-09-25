from __future__ import annotations
import pymysql
import json
from datetime import datetime
from typing import Sequence, Optional, Dict, Any

class MySQLLogger:
    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        self.cfg = dict(host=host, user=user, password=password, database=database, port=port, charset="utf8mb4")

    def _conn(self):
        return pymysql.connect(**self.cfg, cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    def _fetch_metadata(self, track_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """track_metadata에서 필요한 필드(genre, popularity)만 한번에 가져오기"""
        if not track_ids:
            return {}
        qmarks = ",".join(["%s"] * len(track_ids))
        sql = f"SELECT track_id, genre, popularity FROM track_metadata WHERE track_id IN ({qmarks})"
        with self._conn() as con, con.cursor() as cur:
            cur.execute(sql, list(track_ids))
            rows = cur.fetchall()
        return {r["track_id"]: r for r in rows}

    @staticmethod
    def _diversity(returned_ids: Sequence[str]) -> float:
        if not returned_ids:
            return 0.0
        return len(set(returned_ids)) / float(len(returned_ids))

    @staticmethod
    def _avg_popularity(meta_map: Dict[str, Dict[str, Any]], returned_ids: Sequence[str]) -> Optional[float]:
        vals = [meta_map[t]["popularity"] for t in returned_ids if t in meta_map and meta_map[t]["popularity"] is not None]
        return float(sum(vals) / len(vals)) if vals else None

    @staticmethod
    def _genre_precision(meta_map: Dict[str, Dict[str, Any]], seed_ids: Sequence[str], returned_ids: Sequence[str]) -> Optional[float]:
        """간단 버전: 첫 번째 시드의 장르를 기준으로 일치율 계산"""
        if not seed_ids or not returned_ids:
            return None
        seed0 = seed_ids[0]
        if seed0 not in meta_map or meta_map[seed0].get("genre") is None:
            return None
        seed_genre = meta_map[seed0]["genre"]
        hits, total = 0, 0
        for tid in returned_ids:
            g = meta_map.get(tid, {}).get("genre")
            if g is None:
                continue
            total += 1
            if g == seed_genre:
                hits += 1
        if total == 0:
            return None
        return float(hits) / float(total)

    def log_recommend(
        self,
        *,
        by_field: Optional[str],
        query: Optional[str],
        top_k: int,
        elapsed_sec: float,
        seed_track_ids: Sequence[str],
        returned_track_ids: Sequence[str],
    ) -> None:
        """추천 1회에 대해 recommend_logs에 INSERT (품질지표 동시 계산)"""
        # 메타 한번에 가져오기 (seed + returned)
        distinct_ids = list({*seed_track_ids, *returned_track_ids})
        meta_map = self._fetch_metadata(distinct_ids)

        diversity = self._diversity(returned_track_ids)
        avg_popularity = self._avg_popularity(meta_map, returned_track_ids)
        genre_precision = self._genre_precision(meta_map, seed_track_ids, returned_track_ids)

        payload = {
            "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "by_field": by_field,
            "query": query,
            "top_k": top_k,
            "elapsed_sec": float(elapsed_sec),
            "seed_track_ids": json.dumps(list(seed_track_ids), ensure_ascii=False),
            "returned_ids": json.dumps(list(returned_track_ids), ensure_ascii=False),
            "diversity": diversity,
            "avg_popularity": avg_popularity,
            "genre_precision": genre_precision,
        }

        cols = ",".join(payload.keys())
        qmarks = ",".join(["%s"] * len(payload))
        sql = f"INSERT INTO recommend_logs ({cols}) VALUES ({qmarks})"

        with self._conn() as con, con.cursor() as cur:
            cur.execute(sql, list(payload.values()))
