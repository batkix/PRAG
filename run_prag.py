"""
PRAG — Script de démarrage complet
Lance l'entraînement si les modèles n'existent pas, puis démarre l'API.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"

# Limiter les threads OpenBLAS pour économiser la RAM
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def models_exist():
    return (MODELS_DIR / "model_meta.json").exists()

def run_training():
    print("\n🏋️  Lancement de l'entraînement des modèles...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "prag_train.py")],
        cwd=str(BASE_DIR)
    )
    if result.returncode != 0:
        print("❌ L'entraînement a échoué. Vérifiez les logs dans logs/prag_train.log")
        sys.exit(1)
    print("✅ Entraînement terminé !")

def run_api():
    print("\n🌐 Démarrage de l'API PRAG sur http://localhost:5000")
    print("   Appuyez sur Ctrl+C pour arrêter.\n")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    subprocess.run([sys.executable, str(BASE_DIR / "prag_app.py")], cwd=str(BASE_DIR))

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║         PRAG — Démarrage de l'application    ║")
    print("╚══════════════════════════════════════════════╝\n")

    if not models_exist():
        print("⚠️  Aucun modèle trouvé. Entraînement requis.")
        run_training()
    else:
        with open(MODELS_DIR / "model_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"✅ Modèles trouvés :")
        print(f"   CLF : {meta.get('clf_model')}")
        print(f"   REG : {meta.get('reg_model')}")
        print(f"   Entraîné le : {meta.get('trained_at', 'N/A')}")

    run_api()
