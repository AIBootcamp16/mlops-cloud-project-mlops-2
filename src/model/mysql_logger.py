# mysql_logger.py
from __future__ import annotations

import pymysql
import json
from datetime import datetime
from typing import Sequence, Optional, Dict, Any, List
from collections import Counter
import math

class MySQLLogger:
    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        self.cfg = dict(
            host=host, user=user, password=password, database=database,
            port=port, charset="utf8mb4"
        )

    def _conn(self):
        return pymysql.connect(**self.cfg, cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    # --------- 메타 로딩 ----------
    def _fetch_metadata(self, track_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """spotify_music에서 추천 품질 계산에 필요한 필드를 한 번에 로드"""
        track_ids = [str(x).strip() for x in track_ids if x]
        if not track_ids:
            return {}
        qmarks = ",".join(["%s"] * len(track_ids))
        sql = f"""
            SELECT
              track_id, track_name, artist_name, genre, popularity, year
            FROM spotify_music
            WHERE track_id IN ({qmarks})
        """
        with self._conn() as con, con.cursor() as cur:
            cur.execute(sql, list(track_ids))
            rows = cur.fetchall()
        return {str(r["track_id"]).strip(): r for r in rows}

    # --------- 다양성 ----------
    @staticmethod
    def _genre_diversity(meta_map: Dict[str, Dict[str, Any]], returned_ids: Sequence[str]) -> Optional[float]:
        """
        장르 기반 다양성 (고유 장르 수 / 유효 장르가 있는 곡 수).
        장르 정보가 거의 없으면 None.
        """
        genres: List[str] = []
        for tid in returned_ids:
            g = meta_map.get(str(tid).strip(), {}).get("genre")
            if g is not None and str(g).strip() != "":
                genres.append(str(g).strip())
        if not genres:
            return None
        uniq = len(set(genres))
        return float(uniq) / float(len(genres))

    @staticmethod
    def _genre_diversity_simpson(meta_map: Dict[str, Dict[str, Any]], returned_ids: Sequence[str]) -> Optional[float]:
        """
        지니-심프슨 다양성: 1 - Σ p_i^2  (0~1, 높을수록 다양)
        """
        genres: List[str] = []
        for tid in returned_ids:
            g = meta_map.get(str(tid).strip(), {}).get("genre")
            if g is not None and str(g).strip() != "":
                genres.append(str(g).strip())
        if not genres:
            return None
        c = Counter(genres)
        n = sum(c.values())
        p2 = sum((cnt / n) ** 2 for cnt in c.values())
        return 1.0 - p2

    # --------- 평균 인기도 ----------
    @staticmethod
    def _avg_popularity(meta_map: Dict[str, Dict[str, Any]], returned_ids: Sequence[str]) -> Optional[float]:
        vals = []
        for tid in returned_ids:
            v = meta_map.get(str(tid).strip(), {}).get("popularity")
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        return float(sum(vals) / len(vals)) if vals else None

    def _avg_popularity_sql_fallback(self, returned_ids: Sequence[str]) -> Optional[float]:
        returned_ids = [str(x).strip() for x in returned_ids if x]
        if not returned_ids:
            return None
        qmarks = ",".join(["%s"] * len(returned_ids))
        sql = f"SELECT AVG(popularity) AS avg_pop FROM spotify_music WHERE track_id IN ({qmarks})"
        with self._conn() as con, con.cursor() as cur:
            cur.execute(sql, returned_ids)
            row = cur.fetchone()
        return float(row["avg_pop"]) if row and row["avg_pop"] is not None else None

    # --------- 장르 정밀도 ----------
    @staticmethod
    def _genre_precision(meta_map: Dict[str, Dict[str, Any]], seed_ids: Sequence[str], returned_ids: Sequence[str]) -> Optional[float]:
        """
        첫 번째 시드의 장르를 기준으로, returned 중 장르가 같은 비율.
        시드 장르 또는 returned 장르가 없으면 None.
        """
        if not seed_ids or not returned_ids:
            return None
        seed0 = str(seed_ids[0]).strip()
        seed_genre = meta_map.get(seed0, {}).get("genre")
        if seed_genre is None or str(seed_genre).strip() == "":
            return None
        sg = str(seed_genre).strip()

        hits, total = 0, 0
        for tid in returned_ids:
            g = meta_map.get(str(tid).strip(), {}).get("genre")
            if g is None or str(g).strip() == "":
                continue
            total += 1
            if str(g).strip() == sg:
                hits += 1
        if total == 0:
            return None
        return float(hits) / float(total)

    # --------- 로깅 ----------
    def log_recommend(
        self,
        *,
        by_field: Optional[str],
        query: Optional[str],
        top_k: int,
        elapsed_sec: float,
        seed_track_ids: Sequence[str],
        returned_track_ids: Sequence[str],
        distances: Optional[Sequence[float]] = None,  # 있으면 저장
    ) -> None:
        seed_track_ids = [str(x).strip() for x in seed_track_ids if x]
        returned_track_ids = [str(x).strip() for x in returned_track_ids if x]

        # 1) 메타 로딩
        distinct_ids = list({*seed_track_ids, *returned_track_ids})
        meta_map = self._fetch_metadata(distinct_ids)

        # 2) 지표 계산 (장르 기반 다양성으로 교체)
        genre_div = self._genre_diversity(meta_map, returned_track_ids)
        # 필요하면 지니-심프슨 다양성으로 바꿔도 됨:
        # genre_div = self._genre_diversity_simpson(meta_map, returned_track_ids)

        avg_pop = self._avg_popularity(meta_map, returned_track_ids)
        if avg_pop is None:
            avg_pop = self._avg_popularity_sql_fallback(returned_track_ids)  # ✅ 빈값 방어

        gprec = self._genre_precision(meta_map, seed_track_ids, returned_track_ids)

        payload = {
            "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "by_field": by_field,
            "query": query,
            "top_k": int(top_k),
            "elapsed_sec": float(elapsed_sec),
            "seed_track_ids": json.dumps(seed_track_ids, ensure_ascii=False),
            "returned_track_ids": json.dumps(returned_track_ids, ensure_ascii=False),
            "diversity": genre_div,          # ✅ 장르 기반 다양성으로 저장
            "avg_popularity": avg_pop,       # ✅ fallback로 NULL 회피
            "genre_precision": gprec,        # ✅ seed 장르 없으면 None
        }

        cols = ",".join(payload.keys())
        qmarks = ",".join(["%s"] * len(payload))
        sql_log = f"INSERT INTO recommend_logs ({cols}) VALUES ({qmarks})"

        with self._conn() as con, con.cursor() as cur:
            # 3) 로그 저장
            cur.execute(sql_log, list(payload.values()))
            log_id = cur.lastrowid

            # 4) 추천 아이템 저장
            if returned_track_ids:
                sql_item = """
                    INSERT INTO recommend_items
                      (log_id, `rank`, track_id, distance)
                    VALUES (%s,%s,%s,%s)
                """
                rows = []
                for i, tid in enumerate(returned_track_ids, start=1):
                    dist = None
                    if distances and len(distances) >= i and distances[i-1] is not None:
                        try:
                            dist = float(distances[i-1])
                        except Exception:
                            dist = None
                    rows.append((log_id, i, tid, dist))
                cur.executemany(sql_item, rows)
