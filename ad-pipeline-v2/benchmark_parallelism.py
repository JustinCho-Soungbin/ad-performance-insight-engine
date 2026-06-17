"""
benchmark_parallelism.py
─────────────────────────────────────────
Compares sequential (n_jobs=1) vs parallel (n_jobs=-1) training time
for the 5-model benchmark step, using the already-prepared
data/processed/features.csv.

Results are appended to benchmarks/parallelism_results.csv so you
can track this across multiple runs/machines.

Run:
    python benchmark_parallelism.py
"""

import time
import json
import platform
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

FEATURES = [
    "impressions",
    "clicks",
    "CTR",
    "CPC",
    "ad_spend",
    "conversions",
    "CPA",
    "conversion_rate",
    "ctr_cpc_ratio",
    "log_impressions",
    "cost_per_impression",
    "platform_enc",
    "campaign_type_enc",
    "industry_enc",
    "country_enc",
]
TARGET = "high_roas"
RESULTS_PATH = Path("benchmarks/parallelism_results.csv")


def get_models(n_jobs: int):
    """n_jobs=1 -> sequential, n_jobs=-1 -> use all CPU cores."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=n_jobs
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=(1 if n_jobs == 1 else n_jobs),
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
    }


def run_benchmark(X, y, n_jobs: int, label: str) -> float:
    """Runs the same 5-model x 5-fold CV benchmark, returns elapsed seconds."""
    models = get_models(n_jobs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    start = time.perf_counter()
    for name, model in models.items():
        cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "roc_auc"],
            n_jobs=n_jobs,
        )
    elapsed = time.perf_counter() - start

    print(f"[{label}] elapsed: {elapsed:.2f}s")
    return elapsed


def save_result(label: str, n_jobs_setting: str, elapsed: float):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label": label,
                "n_jobs_setting": n_jobs_setting,
                "elapsed_seconds": round(elapsed, 3),
                "cpu_count": __import__("os").cpu_count(),
                "platform": platform.platform(),
            }
        ]
    )
    header = not RESULTS_PATH.exists()
    row.to_csv(RESULTS_PATH, mode="a", header=header, index=False)
    print(f"Saved to {RESULTS_PATH}")


def main():
    df = pd.read_csv("data/processed/features.csv")
    X = df[FEATURES]
    y = df[TARGET]
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features\n")

    # 1. Sequential (n_jobs=1) — original behavior before the fix
    seq_time = run_benchmark(X, y, n_jobs=1, label="sequential")
    save_result("sequential (n_jobs=1)", "1", seq_time)

    # 2. Parallel (n_jobs=-1) — uses all available CPU cores
    par_time = run_benchmark(X, y, n_jobs=-1, label="parallel")
    save_result("parallel (n_jobs=-1)", "-1", par_time)

    # 3. Summary
    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print(f"\n=== Summary ===")
    print(f"Sequential : {seq_time:.2f}s")
    print(f"Parallel   : {par_time:.2f}s")
    print(f"Speedup    : {speedup:.2f}x")

    with open(RESULTS_PATH.parent / "latest_summary.json", "w") as f:
        json.dump(
            {
                "sequential_seconds": round(seq_time, 3),
                "parallel_seconds": round(par_time, 3),
                "speedup": round(speedup, 3),
                "cpu_count": __import__("os").cpu_count(),
            },
            f,
            indent=2,
        )
    print(f"Summary saved to {RESULTS_PATH.parent / 'latest_summary.json'}")


if __name__ == "__main__":
    main()
