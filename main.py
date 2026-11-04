"""Run the MovieLens recommender TUI, rebuild cache, or evaluate the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender import build_demo_data, evaluate_model, load_data, save_demo_data
from tui import print_user, run_tui

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "ml-latest-small"


def print_metrics(result):
    print(f"Training ratings: {result['train_count']:,}")
    print(f"Held-out ratings: {result['test_count']:,}")
    print(f"Validation ratings: {result['validation_count']:,}")
    print("Selected parameters:", json.dumps(result["params"], ensure_ascii=False))
    print("\nModel                         RMSE      MAE")
    for name, metric in result["metrics"].items():
        print(f"{name:<28} {metric['RMSE']:.4f}   {metric['MAE']:.4f}")
    null = result["negative_control"]
    positive = result["positive_control"]
    print(
        f"\nNegative control: {null['repeats']} shuffles; "
        f"RMSE mean={null['RMSE_mean']:.4f} "
        f"[p05={null['RMSE_p05']:.4f}, p95={null['RMSE_p95']:.4f}]"
    )
    print(
        f"Positive control: {'PASS' if positive['passed'] else 'FAIL'}; "
        f"planted prediction {positive['baseline']:.1f} → {positive['prediction']:.1f}"
    )
    print("\nPaired user-bootstrap 95% CI for improvement over bias baseline:")
    for name, interval in result["paired_bootstrap"].items():
        print(
            f"{name:<16} ΔRMSE [{interval['delta_RMSE_95CI'][0]:+.4f}, "
            f"{interval['delta_RMSE_95CI'][1]:+.4f}]  "
            f"ΔMAE [{interval['delta_MAE_95CI'][0]:+.4f}, "
            f"{interval['delta_MAE_95CI'][1]:+.4f}]"
        )


def main():
    parser = argparse.ArgumentParser(description="MovieLens collaborative-filtering recommendation system")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="directory containing ratings.csv and movies.csv")
    parser.add_argument("--evaluate", action="store_true", help="run validation tuning and final held-out evaluation")
    parser.add_argument("--rebuild", action="store_true", help="refit the model and rebuild demo_data.json")
    parser.add_argument("--once", type=int, metavar="USER_ID", help="print one user's recommendations and exit")
    parser.add_argument("--user", type=int, default=66, help="initial user ID for the interactive TUI")
    args = parser.parse_args()
    cache_path = Path(args.data_dir) / "demo_data.json"

    if args.evaluate or args.rebuild:
        ratings, movies = load_data(args.data_dir)
        result = evaluate_model(ratings)
        report_path = Path(args.data_dir) / "evaluation_results.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.evaluate:
            print_metrics(result)
            print(f"\nSaved evaluation details: {report_path}")
        if args.rebuild:
            path = save_demo_data(
                args.data_dir,
                build_demo_data(ratings, movies, params=result["params"], metrics=result["metrics"]),
            )
            print(f"Updated {path}")
        return

    if not cache_path.exists():
        parser.error("demo_data.json not found; run with --rebuild first")
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    if args.once is not None:
        print_user(data, args.once)
    else:
        run_tui(data, initial_user=args.user)


if __name__ == "__main__":
    main()
