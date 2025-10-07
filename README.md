# Q-Learning GridWorld avec Obstacles et Goals Dynamiques

## 🎯 Objectif

Ce projet explore l'influence de **paramètres dynamiques** dans un environnement **GridWorld** sur la performance d'un agent utilisant **Q-Learning**.  
L’agent apprend à atteindre des **goals** tout en évitant des **obstacles** générés de manière dynamique.

---

## 🧩 Description du code (`test_agents.py`)

1. **Paramètres principaux**
   - `sizes` : tailles de grille testées (3x3 à 10x10)
   - `episodes_per_size` : nombre d’épisodes par taille
   - `start` : position de départ `(0,0)`
   - `OBSTACLE_DENSITY` : densité d’obstacles (ex : 0.15 = 15%)
   - `NUM_GOALS` : nombre de goals (1 à 4)
   - `RANDOM_SEED` : seed pour reproductibilité

2. **Génération dynamique**
   - `generate_dynamic_obstacles(size, density, seed)` : crée des obstacles aléatoires proportionnels à la taille de la grille.
   - `generate_dynamic_goals(size, num_goals)` : place les goals selon des positions stratégiques.

3. **Boucle d’entraînement**
   - Pour chaque taille de grille :
     - Création de l’environnement `MyGridWorld` avec obstacles et goals.
     - Initialisation de l’agent Q-Learning `QLearningAgent`.
     - Exécution de `episodes_per_size` épisodes :
       - L’agent choisit des actions (epsilon-greedy)
       - Mise à jour de la Q-table avec `update()`
       - Stockage du retour cumulé par épisode

4. **Visualisations**
   - Courbes individuelles de retour cumulé pour chaque taille
   - Graphiques globaux : moyenne ± écart-type, nombre d’obstacles, courbes normalisées
   - Moyenne mobile pour analyse de convergence
   - Figures sauvegardées dans `figures_dynamic/`
   - Création d’un GIF `performance_dynamic.gif`
   - Sauvegarde CSV `cumulative_rewards_dynamic.csv`

5. **Résultats finaux**
   - Retour moyen et écart-type pour les 10 derniers épisodes
   - Analyse complète : `analyse_complete_dynamic.png`

---

## 📂 Structure du projet

.
├── agents/ # Contient l’implémentation QLearningAgent
├── figures_dynamic/ # Figures individuelles par taille de grille
├── pycache/ # Cache Python
├── grid_3x3.png
├── grid_4x4.png
├── grid_5x5.png
├── grid_6x6.png
├── grid_7x7.png
├── grid_8x8.png
├── grid_9x9.png
├── grid_10x10.png
├── resultat/ # Dossier pour résultats finaux (optionnel)
├── analyse_complete_dynamic.png
├── cumulative_rewards_dynamic.csv
├── performance_dynamic.gif
├── gridworld.py # Définition de l’environnement GridWorld
├── test_agents.py # Script principal d’entraînement et d’analyse
└── README.md

yaml
Copier le code

---

## ⚙️ Instructions d’exécution

1. Créer un environnement virtuel (optionnel mais recommandé) :

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Installer les dépendances :

bash
Copier le code
pip install numpy matplotlib imageio pillow
Exécuter le script principal :

bash
Copier le code
python test_agents.py
Résultats générés :

Figures individuelles : figures_dynamic/grid_<size>x<size>.png

Graphique global complet : analyse_complete_dynamic.png

CSV des récompenses : cumulative_rewards_dynamic.csv

Animation GIF : performance_dynamic.gif

📊 Analyse et interprétation
La performance de l’agent dépend de la taille de la grille et de la densité d’obstacles.

Les courbes cumulées et courbes normalisées permettent de visualiser l’apprentissage et la convergence.

Le GIF offre une visualisation animée de l’évolution des performances pour chaque taille de grille.

Le CSV contient toutes les récompenses cumulées par épisode pour analyses complémentaires.

✅ Notes
Tous les obstacles et goals sont générés dynamiquement à chaque taille de grille, avec seed pour reproductibilité.

Les figures et GIF facilitent la comparaison des performances selon les paramètres de l’environnement.

Les résultats finaux affichent les moyennes et écarts-types des 10 derniers épisodes, pour chaque taille de grille.
