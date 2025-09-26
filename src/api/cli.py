from __future__ import annotations

import argparse
import time

from ..model.model import Recommender
from ..model.logger import RecoLogger, RecommendLog

def _cli_fit(args):
    rec = Recommender(n_neighbors=args.n_neighbors, metric=args.metric).fit_from_csv(args.csv, model_dir=args.model_dir)
    print(f"Fitted and saved to {args.model_dir}")

def _cli_rec(args):
    rec = Recommender.load(args.model_dir)
    t0 = time.time()
    out = rec.recommend(by=args.by, query=args.q, top_k=args.top_k)
    elapsed_sec = time.time() - t0

    seed_ids = rec.artifacts.id_index.iloc[
        rec._lookup_indices(by=args.by, query=args.q)
    ]["track_id"].tolist()[:10]

    logger = RecoLogger()
    logger.log_recommend(RecommendLog(
        by_field=args.by, query=args.q, top_k=args.top_k,
        elapsed_sec=elapsed_sec, seed_track_ids=seed_ids,
        returned_track_ids=out["track_id"].tolist()
    ))
    print(out.to_string(index=False))

def build_argparser():
    p = argparse.ArgumentParser(description="Spotify-style content-based recommender")
    sub = p.add_subparsers(required=True)

    pf = sub.add_parser("fit", help="Fit model from CSV")
    pf.add_argument("--csv", required=True)
    pf.add_argument("--model_dir", default="./models")
    pf.add_argument("--n_neighbors", type=int, default=50)
    pf.add_argument("--metric", default="cosine", choices=["cosine","euclidean","manhattan"])
    pf.set_defaults(func=_cli_fit)

    pr = sub.add_parser("rec", help="Get recommendations")
    pr.add_argument("--model_dir", default="./models")
    pr.add_argument("--by", default="track_name", choices=["track_id","track_name","artist_name"])
    pr.add_argument("--q", required=True)
    pr.add_argument("--top_k", type=int, default=10)
    pr.set_defaults(func=_cli_rec)

    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
