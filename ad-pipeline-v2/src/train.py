"""
train.py
─────────────────────────────────────────
Stage 3: Model Selection & Training
Mirrors notebook cells [12], [14]-[16], [18]-[21]:

1. Benchmark 5 models with 5-fold stratified CV (ROC-AUC)
2. Logistic Regression detailed eval + data leakage check (drop CPA)
3. XGBoost hyperparameter tuning with Optuna (100 trials)
4. Final comparison — save the winner (Logistic Regression, ROC-AUC 0.8777)

Run standalone:
    python src/train.py
"""

import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

INPUT_PATH = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
SCALER_PATH = MODEL_DIR / "scaler.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.csv"

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
N_OPTUNA_TRIALS = 100


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def benchmark_5_models(X_train, y_train) -> dict:
    """Notebook cell [12]: 5-fold stratified CV across 5 candidate models.
    Uses n_jobs=-1 to parallelize across all CPU cores, and cross_validate
    (instead of two separate cross_val_score calls) to avoid running the
    5-fold split twice per model."""
    from sklearn.model_selection import cross_validate

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    logger.info("=== 5-Fold Cross Validation ===")
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "roc_auc"],
            n_jobs=-1,  # run the 5 folds in parallel across CPU cores
        )
        acc = scores["test_accuracy"].mean()
        auc = scores["test_roc_auc"].mean()
        results[name] = {"accuracy": acc, "auc": auc}
        logger.info(f"{name:<25} acc={acc:.4f}  auc={auc:.4f}")

    return results


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Notebook cell [14]: detailed LR eval with scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Logistic Regression test ROC-AUC: {auc:.4f}")
    logger.info(
        f"\n{classification_report(y_test, model.predict(X_test_scaled), target_names=['Low ROAS', 'High ROAS'])}"
    )

    return model, scaler, auc


def check_data_leakage(df: pd.DataFrame, y: pd.Series):
    """Notebook cell [15]: CPA = ad_spend / conversions is correlated with ROAS.
    Verify the model isn't 'cheating' by relying on it."""
    features_no_cpa = [f for f in FEATURES if f != "CPA"]
    X_no_cpa = df[features_no_cpa]

    X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(
        X_no_cpa, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_nc = StandardScaler()
    X_train_nc_scaled = scaler_nc.fit_transform(X_train_nc)
    X_test_nc_scaled = scaler_nc.transform(X_test_nc)

    model_nc = LogisticRegression(max_iter=1000, random_state=42)
    model_nc.fit(X_train_nc_scaled, y_train_nc)
    auc_nc = roc_auc_score(y_test_nc, model_nc.predict_proba(X_test_nc_scaled)[:, 1])

    logger.info(f"Leakage check — ROC-AUC without CPA: {auc_nc:.4f}")
    return auc_nc


def tune_xgboost_optuna(X_train, y_train, n_trials: int = N_OPTUNA_TRIALS):
    """Notebook cell [18]: Bayesian hyperparameter search via Optuna.
    n_jobs=-1 on the study runs multiple trials concurrently across
    CPU cores instead of one at a time."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
            "n_jobs": 1,  # keep per-trial training single-threaded;
            # parallelism comes from running trials concurrently instead
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=-1)

    logger.info(f"Optuna best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Optuna best params: {study.best_params}")
    return study.best_params, study.best_value


def run():
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded feature table: {df.shape[0]} rows")

    X_train, X_test, y_train, y_test = split_data(df)
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 1. Benchmark all 5 models
    cv_results = benchmark_5_models(X_train, y_train)

    # 2. Logistic Regression — detailed eval (expected winner)
    lr_model, scaler, lr_auc = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    # 3. Data leakage check (drop CPA, retrain)
    leakage_auc = check_data_leakage(df, df[TARGET])
    logger.info(
        f"Leakage delta: {abs(lr_auc - leakage_auc):.4f} (small delta = CPA is genuine signal, not a shortcut)"
    )

    # 4. XGBoost + Optuna tuning
    best_params, xgb_cv_auc = tune_xgboost_optuna(
        X_train, y_train, n_trials=N_OPTUNA_TRIALS
    )
    best_params.update({"random_state": 42, "eval_metric": "logloss", "verbosity": 0})
    xgb_tuned = xgb.XGBClassifier(**best_params)
    xgb_tuned.fit(X_train, y_train)
    xgb_test_auc = roc_auc_score(y_test, xgb_tuned.predict_proba(X_test)[:, 1])
    logger.info(f"XGBoost tuned test ROC-AUC: {xgb_test_auc:.4f}")

    # 5. Pick the winner — based on notebook findings, Logistic Regression wins
    final_results = {"Logistic Regression": lr_auc, "XGBoost (tuned)": xgb_test_auc}
    winner_name = max(final_results, key=final_results.get)
    winner_model = lr_model if winner_name == "Logistic Regression" else xgb_tuned

    logger.info(
        f"\n=== Final Winner: {winner_name} (ROC-AUC = {final_results[winner_name]:.4f}) ==="
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_model, BEST_MODEL_PATH)
    joblib.dump(
        scaler, SCALER_PATH
    )  # only meaningful if winner needs scaling (LR does)
    pd.Series(final_results, name="roc_auc").sort_values(ascending=False).to_csv(
        METRICS_PATH
    )

    logger.info(f"Saved best model ({winner_name}) to {BEST_MODEL_PATH}")
    return final_results, winner_name


if __name__ == "__main__":
    run()
