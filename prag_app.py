import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

"""
PRAG — Serveur d'inférence Flask
Expose une API REST pour prédire :
  - La période optimale de semis (classification)
  - Le rendement attendu (régression)
  - Génère un rapport de conséquences pour tout choix de période
"""

import json
import logging
import math
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
from services.weather import WeatherService

# ── Chemins de base ──────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ── Serialization helper ─────────────────────────────────────
def to_serializable(val):
    """Convertit récursivement les types NumPy, Pandas et Path en types Python natifs pour JSON."""
    if isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, (np.ndarray, pd.Series)):
        return [to_serializable(v) for v in val.tolist()]
    elif isinstance(val, pd.DataFrame):
        return val.to_dict(orient="records")
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [to_serializable(v) for v in val]
    elif isinstance(val, Path):
        return str(val)
    elif isinstance(val, datetime):
        return val.isoformat()
    elif isinstance(val, float):
        if np.isnan(val) or np.isneginf(val) or np.isposinf(val):
            return None
        return val
    return val


class SafeJSONProvider(DefaultJSONProvider):
    """Garantit que Flask ne lève aucune TypeError lors de la sérialisation JSON."""
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# ── Application Flask ────────────────────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.json_provider_class = SafeJSONProvider
app.json = SafeJSONProvider(app)
weather_service = WeatherService(LOGS_DIR / "weather_cache.sqlite3")


# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "prag_api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("PRAG-API")


# ── Chargement des modèles ────────────────────────────────────
meta_path = MODELS_DIR / "model_meta.json"
clf_model = None
reg_model = None
feature_names = []


def load_models() -> bool:
    """Charge les modèles de classification et de régression ainsi que les métadonnées."""
    global clf_model, reg_model, feature_names
    if not meta_path.exists():
        log.warning("model_meta.json introuvable. Lancez prag_train.py d'abord.")
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Résolution robuste des chemins de modèles (portabilité multi-machines)
        clf_p = Path(meta.get("clf_model_path", ""))
        if not clf_p.exists():
            clf_p = MODELS_DIR / clf_p.name

        reg_p = Path(meta.get("reg_model_path", ""))
        if not reg_p.exists():
            reg_p = MODELS_DIR / reg_p.name

        if not clf_p.exists() or not reg_p.exists():
            log.error(f"Fichiers modèles introuvables : CLF={clf_p} (existe: {clf_p.exists()}), REG={reg_p} (existe: {reg_p.exists()})")
            return False

        clf_model = joblib.load(clf_p)
        reg_model = joblib.load(reg_p)
        feature_names = list(meta.get("feature_names", []))
        log.info(f"Modèles chargés avec succès : CLF={meta.get('clf_model')} | REG={meta.get('reg_model')}")
        return True
    except Exception as e:
        log.error(f"Erreur lors du chargement des modèles : {e}", exc_info=True)
        return False


# Tentative initiale de chargement
load_models()


def get_clf_probability(model, fv: pd.DataFrame) -> float:
    """Extrait de manière sûre la probabilité de la classe positive (1)."""
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(fv)[0]
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    return float(probas[classes.index(1)])
            if len(probas) > 1:
                return float(probas[1])
            return float(probas[0])
        except Exception:
            pass
    try:
        return float(model.predict(fv)[0])
    except Exception:
        return 0.0


CAT_COLS = ["region", "culture", "type_sol", "saison"]


def normalize_inputs(params: dict) -> dict:
    """Normalise les textes saisis (espaces, variantes d'orthographe)."""
    p = dict(params)
    if "culture" in p:
        cult = str(p["culture"]).strip()
        if cult.lower() in ["mais", "maïs"]:
            cult = "Maïs"
        elif cult.lower() == "cacao":
            cult = "Cacao"
        elif cult.lower() == "anacarde":
            cult = "Anacarde"
        elif cult.lower() == "manioc":
            cult = "Manioc"
        p["culture"] = cult

    if "region" in p:
        p["region"] = str(p["region"]).strip()

    if "type_sol" in p:
        p["type_sol"] = str(p["type_sol"]).strip()

    return p


def build_feature_vector(params: dict) -> pd.DataFrame:
    """Reconstruit le vecteur de features aligné avec les données d'entraînement."""
    params = normalize_inputs(params)
    mois = int(params["mois"])
    region = params["region"]

    # Encodage cyclique
    sin_mois = round(np.sin(2 * np.pi * mois / 12), 4)
    cos_mois = round(np.cos(2 * np.pi * mois / 12), 4)

    # Saison automatique selon la région
    if params.get("saison") in ["Pluies", "Seche"]:
        saison = params["saison"]
    elif region in ["Haut-Sassandra", "Nawa"]:
        saison = "Pluies" if mois in [5, 6, 7, 9, 10] else "Seche"
    else:
        saison = "Pluies" if mois in [6, 7, 8, 9] else "Seche"

    jour_annee_default = int(round((mois - 0.5) * 30.4))
    jour_annee = int(params.get("jour_annee") or jour_annee_default)

    row = {
        "annee": int(params.get("annee", 2025)),
        "mois": mois,
        "trimestre": (mois - 1) // 3 + 1,
        "jour_annee": jour_annee,
        "sin_mois": sin_mois,
        "cos_mois": cos_mois,
        "pluviometrie_mm": float(params["pluviometrie_mm"]),
        "temperature_celsius": float(params["temperature_celsius"]),
        "humidite_pct": float(params["humidite_pct"]),
        "region": region,
        "culture": params["culture"],
        "type_sol": params["type_sol"],
        "saison": saison,
    }

    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False, dtype=int)

    # Réalignement direct et propre avec les colonnes attendues par le modèle
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    return df


def compute_consequences(params: dict, predicted_semis: int, predicted_rendement: float, current_proba: float = None) -> dict:
    """
    Génère un rapport de conséquences pour le choix de période de l'utilisateur.
    Analyse tous les mois de l'année avec le profil climatique régional réaliste et évalue les risques.
    """
    params = normalize_inputs(params)

    mois_labels = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }

    culture_rendements_base = {
        "Cacao": 1.1, "Anacarde": 0.75, "Manioc": 11.5, "Maïs": 2.4
    }

    # Profils climatiques régionaux de référence en Côte d'Ivoire
    CLIMAT_REF = {
        "Haut-Sassandra": {
            "pluies": [5, 6, 7, 9, 10], "transition": [4, 8],
            "pluvio_pluie": 190.0, "temp_pluie": 26.5, "hum_pluie": 86.0,
            "pluvio_trans": 130.0, "temp_trans": 27.5, "hum_trans": 78.0,
            "pluvio_sec": 45.0,    "temp_sec": 30.0,   "hum_sec": 68.0,
        },
        "Nawa": {
            "pluies": [5, 6, 7, 9, 10], "transition": [4, 8],
            "pluvio_pluie": 195.0, "temp_pluie": 26.0, "hum_pluie": 88.0,
            "pluvio_trans": 135.0, "temp_trans": 27.0, "hum_trans": 80.0,
            "pluvio_sec": 50.0,    "temp_sec": 29.5,   "hum_sec": 70.0,
        },
        "Poro": {
            "pluies": [6, 7, 8, 9], "transition": [5, 10],
            "pluvio_pluie": 165.0, "temp_pluie": 28.0, "hum_pluie": 76.0,
            "pluvio_trans": 90.0,  "temp_trans": 30.0, "hum_trans": 60.0,
            "pluvio_sec": 12.0,    "temp_sec": 33.5,   "hum_sec": 42.0,
        }
    }

    mois_actuel = max(1, min(12, int(params["mois"])))
    culture = params["culture"]
    region = params["region"]
    base_rend = culture_rendements_base.get(culture, 2.0)
    ref = CLIMAT_REF.get(region, CLIMAT_REF["Haut-Sassandra"])

    # Prédire pour tous les mois de l'année
    all_months_analysis = []
    for m in range(1, 13):
        p = dict(params)
        p["mois"] = m
        if m != mois_actuel:
            # Pour les autres mois, adapter les conditions météo au climat typique de la région
            if m in ref["pluies"]:
                p["pluviometrie_mm"] = ref["pluvio_pluie"]
                p["temperature_celsius"] = ref["temp_pluie"]
                p["humidite_pct"] = ref["hum_pluie"]
            elif m in ref["transition"]:
                p["pluviometrie_mm"] = ref["pluvio_trans"]
                p["temperature_celsius"] = ref["temp_trans"]
                p["humidite_pct"] = ref["hum_trans"]
            else:
                p["pluviometrie_mm"] = ref["pluvio_sec"]
                p["temperature_celsius"] = ref["temp_sec"]
                p["humidite_pct"] = ref["hum_sec"]

        try:
            if m == mois_actuel and current_proba is not None:
                clf_proba = current_proba
                rend = max(0.0, float(predicted_rendement))
            else:
                fv = build_feature_vector(p)
                clf_proba = get_clf_probability(clf_model, fv)
                rend = max(0.0, float(reg_model.predict(fv)[0]))
        except Exception as e:
            log.debug(f"Erreur prédiction mois {m} : {e}")
            clf_proba = 0.0
            rend = base_rend

        all_months_analysis.append({
            "mois": m,
            "nom_mois": mois_labels[m],
            "probabilite_optimal": round(float(clf_proba), 3),
            "rendement_predit": round(float(rend), 2),
            "est_optimal": bool(clf_proba >= 0.5)
        })

    # Meilleur mois conseillé
    best_month = max(all_months_analysis, key=lambda x: (x["probabilite_optimal"], x["rendement_predit"]))
    optimal_months = [x for x in all_months_analysis if x["est_optimal"]]

    # Évaluer le risque du choix actuel
    current = all_months_analysis[mois_actuel - 1]
    risk_level = "ÉLEVÉ" if current["probabilite_optimal"] < 0.3 else \
                 "MODÉRÉ" if current["probabilite_optimal"] < 0.5 else "FAIBLE"

    rendement_loss_pct = 0.0
    if best_month["rendement_predit"] > 0:
        loss = (best_month["rendement_predit"] - current["rendement_predit"]) / best_month["rendement_predit"] * 100.0
        rendement_loss_pct = round(max(0.0, loss), 1)

    # Conséquences spécifiques selon culture et risque
    consequences = []
    if risk_level in ["ÉLEVÉ", "MODÉRÉ"]:
        consequences.append(f"⚠️  Période sous-optimale pour {culture} dans la région {region}")
        if rendement_loss_pct > 0:
            consequences.append(f"📉  Perte de rendement estimée : -{rendement_loss_pct}% vs période optimale")

        if culture == "Cacao":
            consequences.append("🌧️  Le Cacao nécessite une humidité élevée (>75%) et températures entre 23-29°C")
            consequences.append("🦠  Risque accru de maladies fongiques hors saison des pluies")
        elif culture == "Anacarde":
            consequences.append("☀️  L'Anacarde préfère la saison sèche pour la floraison (région Poro)")
            consequences.append("💧  Excès d'humidité peut provoquer des maladies des fleurs")
        elif culture == "Manioc":
            consequences.append("🌱  Le Manioc tolère bien les variations mais préfère un début de saison des pluies")
            consequences.append("⏱️  Cycle de croissance 8-12 mois — timing crucial pour la récolte")
        elif culture == "Maïs":
            consequences.append("💧  Le Maïs est très sensible au stress hydrique pendant la floraison")
            consequences.append("🌡️  Températures >35°C réduisent significativement la pollinisation")

        if risk_level == "ÉLEVÉ":
            consequences.append(f"🚨  RISQUE ÉLEVÉ : Probabilité de succès seulement {int(round(current['probabilite_optimal'] * 100))}%")
            consequences.append("💡  Recommandation : Reporter au mois optimal ou investir dans l'irrigation")
    else:
        consequences.append(f"✅  Bonne période pour {culture} dans la région {region}")
        consequences.append(f"📈  Rendement estimé : {current['rendement_predit']} tonnes/hectare")

    return {
        "mois_choisi": mois_actuel,
        "nom_mois_choisi": mois_labels[mois_actuel],
        "periode_optimale": bool(predicted_semis == 1),
        "probabilite_succes": current["probabilite_optimal"],
        "niveau_risque": risk_level,
        "rendement_predit": current["rendement_predit"],
        "meilleur_mois": best_month,
        "mois_optimaux": [m["nom_mois"] for m in optimal_months],
        "perte_rendement_estimee_pct": rendement_loss_pct if risk_level != "FAIBLE" else 0.0,
        "consequences": consequences,
        "analyse_annuelle": all_months_analysis
    }


# ── MIDDLEWARE CORS ──────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# ── ROUTES STATIQUES ─────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/<path:path>")
def serve_root_static(path):
    """Permet de servir style.css, app.js et autres fichiers directement depuis la racine."""
    file_path = STATIC_DIR / path
    if file_path.is_file():
        return send_from_directory(str(STATIC_DIR), path)
    return jsonify({"error": f"Fichier introuvable : {path}"}), 404


# ── ROUTES API ────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    """Endpoint principal : retourne prédiction + rapport de conséquences."""
    if clf_model is None or reg_model is None:
        return jsonify({"error": "Modèles non chargés. Lancez prag_train.py d'abord."}), 503

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Corps JSON valide requis"}), 400

    required = ["mois", "region", "culture", "type_sol", "pluviometrie_mm", "temperature_celsius", "humidite_pct"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Champs manquants : {missing}"}), 400

    # Validation et conversion des types d'entrée
    try:
        mois = int(data["mois"])
        if not (1 <= mois <= 12):
            return jsonify({"error": "Le champ 'mois' doit être compris entre 1 et 12"}), 400
        data["mois"] = mois
    except (ValueError, TypeError):
        return jsonify({"error": "Le champ 'mois' doit être un entier valide (1-12)"}), 400

    try:
        data["pluviometrie_mm"] = float(data["pluviometrie_mm"])
        data["temperature_celsius"] = float(data["temperature_celsius"])
        data["humidite_pct"] = float(data["humidite_pct"])
    except (ValueError, TypeError):
        return jsonify({"error": "Les valeurs météo (pluviometrie_mm, temperature_celsius, humidite_pct) doivent être des nombres valides"}), 400

    data = normalize_inputs(data)

    try:
        fv = build_feature_vector(data)
        semis_pred = int(clf_model.predict(fv)[0])
        semis_proba = get_clf_probability(clf_model, fv)
        rendement_pred = max(0.0, float(reg_model.predict(fv)[0]))

        rapport = compute_consequences(data, semis_pred, rendement_pred, current_proba=semis_proba)

        result = {
            "prediction": {
                "periode_semis_optimale": int(semis_pred),
                "probabilite_optimal": round(float(semis_proba), 3),
                "rendement_predit_tonnes_ha": round(float(rendement_pred), 2),
                "conseil": "✅ Période favorable au semis" if semis_pred == 1 else "⚠️ Période non optimale"
            },
            "rapport_consequences": rapport,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "region": data["region"],
                "culture": data["culture"],
                "mois": data["mois"]
            }
        }
        result = to_serializable(result)

        # Journalisation sécurisée de la prédiction dans l'historique
        try:
            log_entry = {
                "timestamp": result["metadata"]["timestamp"],
                "input": data,
                "output": result["prediction"],
                "risk_level": result["rapport_consequences"]["niveau_risque"]
            }
            history_file = LOGS_DIR / "prag_history.json"
            history = {}
            if history_file.exists() and history_file.stat().st_size > 0:
                with open(history_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        history = loaded
            history.setdefault("predictions", []).append(log_entry)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(to_serializable(history), f, indent=2, ensure_ascii=False, default=str)
        except Exception as log_err:
            log.warning(f"Impossible d'écrire l'historique : {log_err}")

        return jsonify(result)

    except Exception as e:
        log.error(f"Erreur prédiction : {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": bool(clf_model is not None and reg_model is not None),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/geocode", methods=["GET"])
def geocode():
    query = (request.args.get("q") or "").strip()
    if len(query) < 2 or len(query) > 100:
        return jsonify({"error": "Saisissez au moins deux caractères (100 maximum)."}), 400
    try:
        return jsonify({"results": weather_service.search_places(query)})
    except Exception:
        log.exception("Échec de la recherche de localité")
        return jsonify({"error": "Recherche de localité momentanément indisponible."}), 503


@app.route("/api/weather", methods=["GET"])
def weather():
    try:
        lat, lon = float(request.args["latitude"]), float(request.args["longitude"])
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Coordonnées de localisation invalides."}), 400
    try:
        return jsonify(weather_service.forecast(lat, lon))
    except Exception:
        log.exception("Échec de récupération météo")
        return jsonify({"error": "Données météo temporairement indisponibles."}), 503


@app.route("/api/history", methods=["GET"])
def get_history():
    history_file = LOGS_DIR / "prag_history.json"
    if not history_file.exists() or history_file.stat().st_size == 0:
        return jsonify({"sessions": [], "best_models": {}, "predictions": []})
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {"sessions": [], "best_models": {}, "predictions": []}
            return jsonify(to_serializable(data))
    except Exception as e:
        log.error(f"Erreur lecture historique : {e}")
        return jsonify({"sessions": [], "best_models": {}, "predictions": []})


@app.route("/api/reload-models", methods=["POST"])
def reload_models():
    success = load_models()
    return jsonify({
        "success": bool(success),
        "message": "Modèles rechargés avec succès" if success else "Échec du rechargement des modèles"
    })


if __name__ == "__main__":
    log.info("🌾 Démarrage du serveur PRAG...")
    app.run(debug=True, host="0.0.0.0", port=5000)
