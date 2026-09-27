# 🌾 PRAG — Prédiction Agricole par Intelligence Artificielle

> **Prédire les meilleures périodes de culture en Côte d'Ivoire**

PRAG utilise deux modèles d'IA entraînés sur des données climatiques ivoiriennes pour :
- **Classifier** si un mois est optimal pour le semis (`periode_semis_optimale`)  
- **Prédire** le rendement attendu en tonnes/hectare (`rendement_tonnes_ha`)

---

## 📁 Structure du projet

```
PRAG/
├── 📊 Données
│   ├── agripredict_dataset.csv   # Dataset complet (10 000 lignes)
│   ├── train.csv                  # Entraînement 2021–2023 (7 438 lignes)
│   └── test.csv                   # Évaluation 2024 (2 562 lignes)
│
├── 🤖 Modèles & Entraînement
│   ├── prag_train.py              # Pipeline d'entraînement CPU (local)
│   ├── prag_kaggle_gpu.ipynb      # Notebook entraînement GPU (Kaggle)
│   └── models/                    # Modèles sauvegardés (.pkl)
│
├── 🌐 Application Web
│   ├── prag_app.py                # Serveur API Flask
│   ├── run_prag.py                # Script de démarrage tout-en-un
│   └── static/
│       ├── index.html             # Interface utilisateur
│       ├── style.css              # Design premium
│       └── app.js                 # Logique frontend
│
└── 📝 Logs & Historique
    └── logs/
        ├── prag_history.json      # Historique complet (sessions + prédictions)
        └── prag_train.log         # Logs détaillés d'entraînement
```

---

## 🚀 Démarrage rapide

### 1. Installer les dépendances

```bash
pip install pandas numpy scikit-learn joblib xgboost lightgbm flask --no-cache-dir
```

### 2. Entraîner les modèles (CPU local)

```bash
python prag_train.py
```

### 3. Lancer l'application web

```bash
python run_prag.py
# OU directement :
python prag_app.py
```

Ouvrez ensuite → **http://localhost:5000**

---

## ☁️ Entraînement GPU sur Kaggle (réseaux de neurones)

1. Allez sur [kaggle.com](https://www.kaggle.com) → **New Notebook**
2. Uploadez `train.csv` et `test.csv` comme dataset
3. Ouvrez `prag_kaggle_gpu.ipynb` et importez-le sur Kaggle
4. Activez le GPU : **Settings → Accelerator → GPU T4 x2**
5. **Run All**
6. Téléchargez `prag_gpu_history.json` et placez-le dans `logs/`

---

## 🤖 Modèles comparés

### Classification (`periode_semis_optimale`)
| Modèle | Métrique | Vitesse |
|--------|----------|---------|
| LogisticRegression | F1-score | ⚡ Ultra-rapide |
| DecisionTree | F1-score | ⚡ Très rapide |
| RandomForest | F1-score | 🚀 Rapide |
| GradientBoosting | F1-score | 🔄 Moyen |
| **XGBoost** | F1-score | 🚀 Rapide |
| **LightGBM** | F1-score | ⚡ Très rapide |
| MLP GPU | F1-score | ☁️ Kaggle requis |

### Régression (`rendement_tonnes_ha`)
| Modèle | Métrique | Vitesse |
|--------|----------|---------|
| Ridge | RMSE / R² | ⚡ Ultra-rapide |
| DecisionTree | RMSE / R² | ⚡ Très rapide |
| RandomForest | RMSE / R² | 🚀 Rapide |
| **XGBoost** | RMSE / R² | 🚀 Rapide |
| **LightGBM** | RMSE / R² | ⚡ Très rapide |
| MLP GPU | RMSE / R² | ☁️ Kaggle requis |

### Critère de sélection du meilleur modèle
```
Score composite = 0.7 × Performance + 0.3 × (1 - Vitesse_inférence_normalisée)
```
> Priorité à la performance (70%) tout en pénalisant les modèles trop lents (30%).

---

## 📊 Rapport de conséquences

Pour **tout choix de période**, même non conseillé, PRAG génère :
- ✅ / ⚠️ Verdict de la période choisie
- 📉 Perte de rendement estimée vs période optimale
- 🗓️ Liste des mois optimaux par région/culture
- 📅 Calendrier annuel avec probabilités mois par mois
- 🌿 Conseils agronomiques spécifiques à la culture

---

## 📝 Historique complet

Tout est journalisé dans `logs/prag_history.json` :
- Sessions d'entraînement (modèles testés, métriques, durée)
- Prédictions effectuées (paramètres, résultats, niveau de risque)
- Meilleurs modèles actuels

---

## 🌍 Cultures & Régions couvertes

| Culture | Région optimale | Mois favorables |
|---------|----------------|-----------------|
| 🍫 Cacao | Haut-Sassandra, Nawa | Avril, Mai, Août, Septembre |
| 🥜 Anacarde | Poro | Mai, Juin |
| 🌱 Manioc | Haut-Sassandra, Nawa | Avril, Mai, Août, Septembre |
| 🌽 Maïs | Toutes | Selon saison des pluies |

---

*© 2025 PRAG — Intelligence Agricole pour la Côte d'Ivoire 🌍*

## Météo et géolocalisation (module ajouté)

Le tableau météo appelle le service Flask, jamais Open-Meteo directement depuis les composants : `services/weather.py` fournit la recherche de localités et les prévisions. Les réponses météo sont conservées dans `logs/weather_cache.sqlite3` (cache de 30 minutes) et la dernière réponse reste disponible en cas de panne réseau. Les localités enregistrées sont conservées dans le navigateur de l'utilisateur (maximum huit).

Sources utilisées : Open-Meteo Forecast API (conditions actuelles, horaires et prévisions à sept jours) et Open-Meteo Geocoding API (GeoNames). Aucune clé n'est requise pour l'usage non commercial. Open-Meteo annonce jusqu'à 10 000 appels par jour pour l'accès gratuit non commercial ; la disponibilité n'est pas garantie. L'attribution Open-Meteo est affichée dans l'interface. Une utilisation commerciale nécessite une licence adaptée. Les indications agronomiques affichées sont des règles explicites basées sur les cumuls de pluie et la température prévue, pas des prédictions de rendement.

## État fonctionnel et limites

Le projet d'origine reste le simulateur ML de période de semis et rendement, avec modèles joblib pré-entraînés. Le module météo ajoute `/api/geocode?q=...` et `/api/weather?latitude=...&longitude=...`. Pour lancer : `pip install -r requirements.txt`, puis `python prag_app.py`, et ouvrir `http://localhost:5000`.

Ce dépôt n'inclut pas encore de comptes/authentification, profils serveur, exploitations/parcelles, base de connaissances documentée sur les cycles, assistant IA génératif ni génération/planification de rapports. Les localités sont enregistrées localement dans le navigateur, sans séparation par compte. Le simulateur historique continue d'utiliser ses entrées météo saisies par l'utilisateur et son modèle existant ; il ne consomme pas automatiquement la météo live. Il ne faut donc pas présenter ces éléments comme achevés. La recherche de villes est actuellement filtrée sur la Côte d'Ivoire.
