# AgriPredict — Jeu de données synthétique pour la Côte d'Ivoire

## 1. Objectif
Jeu de données synthétique conçu pour entraîner deux modèles de prédiction agricole :
- **Régression** → `rendement_tonnes_ha` (rendement attendu, en tonnes/hectare)
- **Classification binaire** → `periode_semis_optimale` (1 = période favorable au semis, 0 = sinon)

## 2. Fichiers livrés
| Fichier | Lignes | Usage |
|---|---|---|
| `agripredict_dataset.csv` | 10 000 | Jeu complet (exploration, statistiques) |
| `train.csv` | 7 438 (2021–2023) | Entraînement |
| `test.csv` | 2 562 (2024) | Évaluation finale |

**Le split est chronologique, pas aléatoire.** On entraîne sur le passé (2021‑2023) et on
évalue sur l'année la plus récente (2024). C'est la façon correcte de valider un modèle
destiné à prédire l'avenir : un split aléatoire mélangerait les années et donnerait une
estimation de performance trop optimiste (fuite d'information temporelle).

## 3. Colonnes

**Identifiants / temporel**
- `date`, `annee`, `mois`, `trimestre`, `jour_annee` : le texte brut `date` n'est pas
  directement exploitable par un modèle — ne pas l'utiliser comme feature, seules les
  colonnes numériques dérivées le sont.
- `sin_mois`, `cos_mois` : encodage cyclique du mois (évite la fausse distance
  numérique entre décembre et janvier). Utile surtout pour les modèles linéaires /
  réseaux de neurones ; sans effet négatif pour les modèles à arbres.

**Catégorielles** (à encoder en one-hot ou en `category` avant entraînement)
- `region` : Haut-Sassandra, Nawa, Poro
- `culture` : Cacao, Anacarde, Manioc, Maïs
- `type_sol` : Argileux, Limoneux, Sablonneux
- `saison` : Pluies, Seche

**Numériques continues**
- `pluviometrie_mm`, `temperature_celsius`, `humidite_pct`

**Cibles**
- `periode_semis_optimale` (0/1) — classe positive ≈ 31 % du jeu (déséquilibre modéré,
  gérable sans ré-échantillonnage, mais surveiller le F1-score plutôt que la seule
  accuracy)
- `rendement_tonnes_ha` (continue, 0.3 à ~22 selon la culture — forte hétéroscédasticité
  entre Manioc et les autres cultures : envisager un modèle par culture ou une
  transformation log si un seul modèle global est utilisé)

## 4. Points de vigilance pour l'entraînement
- **Ne pas utiliser `rendement_tonnes_ha` comme feature pour prédire
  `periode_semis_optimale`, ni l'inverse** : les deux cibles partagent une dépendance
  causale aux mêmes variables météo, ce qui créerait une fuite artificielle si l'une
  sert d'entrée pour prédire l'autre.
- Un bruit de label de 3 % a été injecté sur `periode_semis_optimale` pour éviter
  qu'un modèle atteigne une accuracy de 100 % en mémorisant simplement la règle
  région+mois — un jeu de données parfaitement déterministe n'a pas de valeur
  pédagogique pour évaluer une vraie capacité de généralisation.
- Les corrélations `culture` × `region` et `saison` × `pluviométrie/température` sont
  volontairement fortes et réalistes (zones forestières vs zones de savane) : c'est le
  signal que le modèle doit apprendre, pas un artefact à corriger.

## 5. Reproductibilité
`RANDOM_SEED = 42` dans `generate_dataset.py`. Relancer le script produit exactement
les mêmes fichiers. Pour un jeu plus volumineux (utile pour un réseau de neurones),
augmenter `N_SAMPLES` en tête de script.
