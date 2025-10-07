1. 🎯 Objectif du projet

Ce projet vise à étudier l’influence de la taille du GridWorld sur la performance d’un agent Q-Learning.
L’environnement GridWorld est une grille carrée où l’agent doit atteindre un objectif en évitant les obstacles.

Pour chaque taille de grille (de 3x3 à 10x10), l’agent est entraîné pendant plusieurs épisodes, et les retours cumulés (récompenses totales par épisode) sont mesurés afin d’analyser comment la complexité de l’environnement affecte l’apprentissage.

2. 🧩 Structure du projet
Mini-RL-GridWorld/
│
├─ agents/
│   └─ q_learning.py          # Définition de l’agent Q-Learning
│
├─ figures/                   # Images générées pour chaque taille de grille
│   ├─ grid_3x3.png
│   ├─ grid_4x4.png
│   └─ ...
│
├─ gridworld.py               # Définition de l’environnement MyGridWorld
├─ test_agents.py             # Script principal d’analyse et de visualisation
├─ cumulative_rewards.csv     # Fichier CSV contenant les retours cumulés
├─ performance_gridworld.gif  # Animation de l’évolution des performances
├─ analyse_complete.png       # Figure récapitulative complète
├─ reward_analyse/            # (Optionnel) Dossier d’analyse complémentaire
├─ venv/                      # Environnement virtuel Python
└─ README.md                  # Ce fichier

3. ⚙️ Installation et setup
Créer un environnement virtuel
python -m venv venv

Activer l’environnement

Windows (PowerShell) :

.\venv\Scripts\Activate.ps1


Windows (cmd) :

.\venv\Scripts\activate.bat


Linux / MacOS :

source venv/bin/activate

Installer les dépendances
pip install gymnasium numpy matplotlib imageio

4. 🚀 Exécution du projet

Assurez-vous que l’environnement virtuel est activé, puis exécutez :

python test_agents.py


Le script :

entraîne un agent Q-Learning sur plusieurs tailles de grilles (de 3x3 à 10x10),

enregistre les retours cumulés dans un fichier CSV (cumulative_rewards.csv),

génère un graphique par taille de grille dans le dossier figures/,

crée un GIF comparatif (performance_gridworld.gif),

produit une figure récapitulative complète (analyse_complete.png) montrant :

l’évolution du retour cumulé par épisode,

la performance moyenne selon la taille,

les courbes normalisées et les performances finales.

5. 📊 Résultats attendus

Plus la grille est grande, plus l’agent met du temps à apprendre une bonne politique.

Le retour moyen diminue généralement avec la taille, à cause d’un espace d’état plus complexe.

Le GIF permet de visualiser la progression de la performance pour chaque taille de grille.

6. 📁 Fichiers générés automatiquement
Fichier / Dossier	Description
figures/	Graphiques de performance par taille
cumulative_rewards.csv	Retours cumulés pour chaque taille
performance_gridworld.gif	Animation de l’évolution des apprentissages
analyse_complete.png	Figure récapitulative finale
reward_analyse/	(Optionnel) Analyses complémentaires
7. 🧠 À propos du projet

Ce projet fait partie d’une exploration des techniques d’apprentissage par renforcement (RL) appliquées à des environnements simples.
Il met en évidence comment la complexité de l’environnement (taille de la grille) influence les performances d’un agent Q-Learning.
