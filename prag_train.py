# ── Limiter la RAM OpenBLAS dès le démarrage (avant tout import numpy) ──
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

"""
╔══════════════════════════════════════════════════════════════════════╗
║          PRAG — Pipeline d'entraînement & sélection de modèle       ║
║          Prédiction des meilleures périodes agricoles (CI)           ║
╚══════════════════════════════════════════════════════════════════════╝

Tâches :
  1. Classification  → periode_semis_optimale (0/1)
  2. Régression      → rendement_tonnes_ha

Stratégie de sélection :
  Meilleur compromis vitesse d'inférence / performance.
  Les modèles GPU (réseaux de neurones) sont traités sur Kaggle.
"""

import json
import time
import warnings
import logging
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost non disponible — ignoré.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] LightGBM non disponible — ignoré.")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"
HISTORY_FILE  = LOGS_DIR / "prag_history.json"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

RANDOM_SEED   = 42
SESSION_START = datetime.now(timezone.utc).isoformat()

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "prag_train.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("PRAG")


# ─────────────────────────────────────────────────────────────
# HISTORIQUE JSON — persistant entre les sessions
# ─────────────────────────────────────────────────────────────
def load_history() -> dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sessions": [], "best_models": {}, "all_runs": []}


def save_history(history: dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Historique sauvegardé → {HISTORY_FILE}")


# ─────────────────────────────────────────────────────────────
# PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────
CAT_COLS = ["region", "culture", "type_sol", "saison"]
DROP_COLS = ["date", "periode_semis_optimale", "rendement_tonnes_ha"]

FEATURE_COLS = [
    "annee", "mois", "trimestre", "jour_annee", "sin_mois", "cos_mois",
    "pluviometrie_mm", "temperature_celsius", "humidite_pct",
    "region", "culture", "type_sol", "saison"
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding des variables catégorielles."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False, dtype=int)
    return df


def prepare_data(train_path: str, test_path: str):
    log.info("─── Chargement des données ───")
    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    log.info(f"Train : {train_raw.shape} | Test : {test_raw.shape}")

    # Cibles
    y_clf_train = train_raw["periode_semis_optimale"].values
    y_reg_train = train_raw["rendement_tonnes_ha"].values
    y_clf_test  = test_raw["periode_semis_optimale"].values
    y_reg_test  = test_raw["rendement_tonnes_ha"].values

    # Features brutes
    X_train_raw = train_raw[FEATURE_COLS].copy()
    X_test_raw  = test_raw[FEATURE_COLS].copy()

    # Encodage (on joint train+test pour s'assurer des mêmes colonnes one-hot)
    X_all = pd.concat([X_train_raw, X_test_raw], axis=0)
    X_all_enc = encode_features(X_all)

    n_train = len(X_train_raw)
    X_train = X_all_enc.iloc[:n_train].reset_index(drop=True)
    X_test  = X_all_enc.iloc[n_train:].reset_index(drop=True)

    log.info(f"Features après encodage : {X_train.shape[1]} colonnes")

    return (X_train, y_clf_train, y_reg_train,
            X_test,  y_clf_test,  y_reg_test,
            list(X_train.columns))


# ─────────────────────────────────────────────────────────────
# DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────────────────────
def get_clf_candidates():
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED),
    }
    if HAS_XGB:
        candidates["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_SEED,
            tree_method="hist", device="cpu", n_jobs=1
        )
    if HAS_LGB:
        candidates["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=1, verbose=-1
        )
    return candidates


def get_reg_candidates():
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED),
    }
    if HAS_XGB:
        candidates["XGBoost"] = xgb.XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, tree_method="hist", device="cpu", n_jobs=1
        )
    if HAS_LGB:
        candidates["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=1, verbose=-1
        )
    return candidates


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────────────────────
def train_and_evaluate_clf(X_train, y_train, X_test, y_test, history):
    log.info("\n══════ CLASSIFICATION : periode_semis_optimale ══════")
    results = []

    candidates = get_clf_candidates()
    for name, model in candidates.items():
        log.info(f"  ▶ Entraînement : {name}")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time_ms = (time.perf_counter() - t1) * 1000 / len(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        f1    = f1_score(y_test, y_pred, average="binary")
        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        log.info(f"     F1={f1:.4f}  Acc={acc:.4f}  AUC={auc_str}  "
                 f"Train={train_time:.2f}s  Inférence={infer_time_ms:.4f}ms/obs")

        run = {
            "task": "classification",
            "model": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": "CPU",
            "metrics": {
                "f1_score": round(f1, 4),
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4) if auc else None,
                "train_time_sec": round(train_time, 3),
                "inference_ms_per_obs": round(infer_time_ms, 5)
            },
            "model_object": model
        }
        results.append(run)
        history["all_runs"].append({k: v for k, v in run.items() if k != "model_object"})

    # Sélection du meilleur modèle : score composite = 0.7*F1 - 0.3*norm(infer_time)
    max_infer = max(r["metrics"]["inference_ms_per_obs"] for r in results)
    min_infer = min(r["metrics"]["inference_ms_per_obs"] for r in results)
    rng = max_infer - min_infer if max_infer != min_infer else 1

    for r in results:
        norm_infer = (r["metrics"]["inference_ms_per_obs"] - min_infer) / rng
        r["composite_score"] = 0.7 * r["metrics"]["f1_score"] - 0.3 * norm_infer

    best = max(results, key=lambda r: r["composite_score"])
    log.info(f"\n  ✅ Meilleur modèle CLF : {best['model']}  "
             f"(composite={best['composite_score']:.4f})")

    # Rapport détaillé du meilleur modèle
    y_pred_best = best["model_object"].predict(X_test)
    log.info("\n" + classification_report(y_test, y_pred_best, target_names=["Non-optimal", "Optimal"]))
    log.info(f"Matrice de confusion :\n{confusion_matrix(y_test, y_pred_best)}")

    return best, results


def train_and_evaluate_reg(X_train, y_train, X_test, y_test, history):
    log.info("\n══════ RÉGRESSION : rendement_tonnes_ha ══════")
    results = []

    candidates = get_reg_candidates()
    for name, model in candidates.items():
        log.info(f"  ▶ Entraînement : {name}")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time_ms = (time.perf_counter() - t1) * 1000 / len(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        log.info(f"     RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
                 f"Train={train_time:.2f}s  Inférence={infer_time_ms:.4f}ms/obs")

        run = {
            "task": "regression",
            "model": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": "CPU",
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2_score": round(r2, 4),
                "train_time_sec": round(train_time, 3),
                "inference_ms_per_obs": round(infer_time_ms, 5)
            },
            "model_object": model
        }
        results.append(run)
        history["all_runs"].append({k: v for k, v in run.items() if k != "model_object"})

    # Sélection : score composite = 0.7*(1-norm_rmse) - 0.3*norm_infer
    max_rmse = max(r["metrics"]["rmse"] for r in results)
    min_rmse = min(r["metrics"]["rmse"] for r in results)
    rng_rmse = max_rmse - min_rmse if max_rmse != min_rmse else 1

    max_infer = max(r["metrics"]["inference_ms_per_obs"] for r in results)
    min_infer = min(r["metrics"]["inference_ms_per_obs"] for r in results)
    rng_infer = max_infer - min_infer if max_infer != min_infer else 1

    for r in results:
        norm_rmse  = (r["metrics"]["rmse"] - min_rmse) / rng_rmse
        norm_infer = (r["metrics"]["inference_ms_per_obs"] - min_infer) / rng_infer
        r["composite_score"] = 0.7 * (1 - norm_rmse) - 0.3 * norm_infer

    best = max(results, key=lambda r: r["composite_score"])
    log.info(f"\n  ✅ Meilleur modèle REG : {best['model']}  "
             f"(composite={best['composite_score']:.4f})")

    return best, results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║          PRAG — Entraînement des modèles         ║")
    log.info(f"║  Session : {SESSION_START}  ║")
    log.info("╚══════════════════════════════════════════════════╝")

    history = load_history()

    session_record = {
        "session_id": SESSION_START,
        "started_at": SESSION_START,
        "status": "running",
        "operations": []
    }

    # ── Chargement ──────────────────────────────────────────
    t_start = time.perf_counter()
    (X_train, y_clf_train, y_reg_train,
     X_test,  y_clf_test,  y_reg_test,
     feature_names) = prepare_data(
         str(BASE_DIR / "train.csv"),
         str(BASE_DIR / "test.csv")
     )

    session_record["operations"].append({
        "op": "data_loading",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": len(feature_names),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Classification ──────────────────────────────────────
    best_clf, clf_results = train_and_evaluate_clf(
        X_train, y_clf_train, X_test, y_clf_test, history
    )

    session_record["operations"].append({
        "op": "classification_training",
        "models_trained": [r["model"] for r in clf_results],
        "best_model": best_clf["model"],
        "best_metrics": best_clf["metrics"],
        "composite_score": round(best_clf["composite_score"], 6),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Régression ───────────────────────────────────────────
    best_reg, reg_results = train_and_evaluate_reg(
        X_train, y_reg_train, X_test, y_reg_test, history
    )

    session_record["operations"].append({
        "op": "regression_training",
        "models_trained": [r["model"] for r in reg_results],
        "best_model": best_reg["model"],
        "best_metrics": best_reg["metrics"],
        "composite_score": round(best_reg["composite_score"], 6),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Sauvegarde des modèles ───────────────────────────────
    log.info("\n─── Sauvegarde des modèles ───")

    clf_path = MODELS_DIR / f"clf_{best_clf['model'].lower()}.pkl"
    reg_path = MODELS_DIR / f"reg_{best_reg['model'].lower()}.pkl"

    joblib.dump(best_clf["model_object"], clf_path)
    joblib.dump(best_reg["model_object"], reg_path)

    # Sauvegarder aussi les noms de features pour l'inférence
    meta = {
        "feature_names": feature_names,
        "cat_cols_used": CAT_COLS,
        "clf_model": best_clf["model"],
        "reg_model": best_reg["model"],
        "clf_model_path": str(clf_path),
        "reg_model_path": str(reg_path),
        "trained_at": SESSION_START
    }
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(f"  CLF sauvegardé → {clf_path}")
    log.info(f"  REG sauvegardé → {reg_path}")
    log.info(f"  Meta  sauvegardé → {meta_path}")

    session_record["operations"].append({
        "op": "model_export",
        "clf_path": str(clf_path),
        "reg_path": str(reg_path),
        "meta_path": str(meta_path),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # ── Clôture de session ───────────────────────────────────
    total_time = time.perf_counter() - t_start
    session_record["status"] = "completed"
    session_record["completed_at"] = datetime.now(timezone.utc).isoformat()
    session_record["total_duration_sec"] = round(total_time, 2)

    history["sessions"].append(session_record)
    history["best_models"] = {
        "classification": {
            "model": best_clf["model"],
            "metrics": best_clf["metrics"],
            "path": str(clf_path),
            "updated_at": SESSION_START
        },
        "regression": {
            "model": best_reg["model"],
            "metrics": best_reg["metrics"],
            "path": str(reg_path),
            "updated_at": SESSION_START
        }
    }

    save_history(history)

    log.info(f"\n╔══════════════════════════════════════════════╗")
    log.info(f"║  Pipeline terminé en {total_time:.1f}s")
    log.info(f"║  CLF → {best_clf['model']} | F1={best_clf['metrics']['f1_score']}")
    log.info(f"║  REG → {best_reg['model']} | RMSE={best_reg['metrics']['rmse']}")
    log.info(f"╚══════════════════════════════════════════════╝")

    return best_clf, best_reg, meta


if __name__ == "__main__":
    main()
