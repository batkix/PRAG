import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SAMPLES = 10000          # volume suffisant pour un entraînement robuste
ADD_LABEL_NOISE = True     # simule les imperfections réelles de terrain
NOISE_RATE = 0.03          # 3% des labels "periode_semis_optimale" sont inversés

regions = ['Haut-Sassandra', 'Nawa', 'Poro']
cultures = ['Cacao', 'Anacarde', 'Manioc', 'Maïs']
types_sol = ['Argileux', 'Limoneux', 'Sablonneux']

start_date = datetime(2021, 1, 1)
dates = [start_date + timedelta(days=int(i)) for i in np.random.randint(0, 1460, size=N_SAMPLES)]

data = []

for i in range(N_SAMPLES):
    date = dates[i]
    month = date.month
    year = date.year

    # 1. Distribution géographique réaliste en Côte d'Ivoire
    region = np.random.choice(regions, p=[0.40, 0.35, 0.25])

    # 2. Adéquation des cultures selon les régions
    if region in ['Haut-Sassandra', 'Nawa']:
        culture = np.random.choice(cultures, p=[0.50, 0.05, 0.30, 0.15])
    else:
        culture = np.random.choice(cultures, p=[0.05, 0.55, 0.15, 0.25])

    # 3. Profil climatique régional et saisonnier
    if region in ['Haut-Sassandra', 'Nawa']:
        saison = 'Pluies' if month in [5, 6, 7, 9, 10] else 'Seche'
        if saison == 'Pluies':
            pluvio = np.random.normal(190, 35)
            temp = np.random.normal(26.5, 1.8)
            hum = np.random.normal(86, 4)
        else:
            pluvio = np.random.normal(45, 20)
            temp = np.random.normal(30.0, 2.2)
            hum = np.random.normal(68, 6)
    else:
        saison = 'Pluies' if month in [6, 7, 8, 9] else 'Seche'
        if saison == 'Pluies':
            pluvio = np.random.normal(165, 30)
            temp = np.random.normal(28.0, 2.0)
            hum = np.random.normal(76, 5)
        else:
            pluvio = np.random.normal(12, 8)
            temp = np.random.normal(33.5, 2.8)
            hum = np.random.normal(42, 8)

    # 4. Sol adapté au terrain
    sol_weights = [0.45, 0.40, 0.15] if region != 'Poro' else [0.20, 0.30, 0.50]
    sol = np.random.choice(types_sol, p=sol_weights)

    # Recadrage physique
    pluvio = max(0.0, round(pluvio, 1))
    temp = max(18.0, min(43.0, round(temp, 1)))
    hum = max(15.0, min(100.0, round(hum, 1)))

    # 5. Règle métier : période optimale de semis (cible classification)
    optimal_semis = 0
    if region in ['Haut-Sassandra', 'Nawa'] and month in [4, 5, 8, 9]:
        optimal_semis = 1
    elif region == 'Poro' and month in [5, 6]:
        optimal_semis = 1

    # Bruit de label réaliste (erreurs de saisie / cas limites terrain)
    if ADD_LABEL_NOISE and np.random.random() < NOISE_RATE:
        optimal_semis = 1 - optimal_semis

    # 6. Calcul du rendement (cible régression : tonnes/hectare)
    base_rendement = {'Cacao': 1.1, 'Anacarde': 0.75, 'Manioc': 11.5, 'Maïs': 2.4}[culture]
    affinite_region = 1.25 if (culture in ['Cacao', 'Manioc'] and region != 'Poro') or (culture == 'Anacarde' and region == 'Poro') else 0.65
    affinite_meteo = 1.18 if (110 <= pluvio <= 230) and (23 <= temp <= 29) else 0.82
    bruit = np.random.normal(1.0, 0.08)
    rendement = base_rendement * affinite_region * affinite_meteo * bruit
    rendement = max(0.1, round(rendement, 2))

    # 7. Features temporelles exploitables par un modèle (le texte brut 'date' ne l'est pas)
    jour_annee = date.timetuple().tm_yday
    trimestre = (month - 1) // 3 + 1
    sin_mois = round(np.sin(2 * np.pi * month / 12), 4)   # encodage cyclique :
    cos_mois = round(np.cos(2 * np.pi * month / 12), 4)   # évite la discontinuité déc→janv

    data.append({
        'date': date.strftime('%Y-%m-%d'),
        'annee': year,
        'mois': month,
        'trimestre': trimestre,
        'jour_annee': jour_annee,
        'sin_mois': sin_mois,
        'cos_mois': cos_mois,
        'region': region,
        'culture': culture,
        'type_sol': sol,
        'saison': saison,
        'pluviometrie_mm': pluvio,
        'temperature_celsius': temp,
        'humidite_pct': hum,
        'periode_semis_optimale': optimal_semis,
        'rendement_tonnes_ha': rendement
    })

df = pd.DataFrame(data)

# ============================================================
# CONTRÔLES QUALITÉ
# ============================================================
assert df.isnull().sum().sum() == 0, "Valeurs manquantes détectées"
assert N_SAMPLES == len(df), "Nombre de lignes incohérent"

# ============================================================
# EXPORTS
# ============================================================
df.to_csv('agripredict_dataset.csv', index=False)

# Découpage CHRONOLOGIQUE (et non aléatoire) : on entraîne sur le passé (2021-2023)
# et on évalue sur l'année la plus récente (2024). C'est la façon correcte de valider
# un modèle de prédiction destiné à être utilisé sur des données futures : un split
# aléatoire mélangerait les années et donnerait une estimation trop optimiste.
train_df = df[df['annee'] <= 2023].reset_index(drop=True)
test_df = df[df['annee'] == 2024].reset_index(drop=True)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# ============================================================
# RAPPORT DE CONTRÔLE (console)
# ============================================================
print(f"Dataset complet : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Train (2021-2023) : {train_df.shape[0]} lignes ({train_df.shape[0]/df.shape[0]:.1%})")
print(f"Test  (2024)      : {test_df.shape[0]} lignes ({test_df.shape[0]/df.shape[0]:.1%})")
print("\nÉquilibre de la cible de classification 'periode_semis_optimale' :")
print(df['periode_semis_optimale'].value_counts(normalize=True).round(3))
print("\nStatistiques de la cible de régression 'rendement_tonnes_ha' :")
print(df['rendement_tonnes_ha'].describe().round(2))
print("\nRépartition par culture :")
print(df['culture'].value_counts(normalize=True).round(3))
