README – Mini RL GridWorld
1. Objectif du projet

Le projet consiste à créer un environnement GridWorld simple et à tester différents agents de Reinforcement Learning (RL) sur cet environnement.

Environnement : GridWorld 5x5 avec un état de départ et un état objectif.

Agents à implémenter :

Policy Iteration

Value Iteration

Monte Carlo (First-Visit)

Q-Learning

Le but est de comparer les politiques et les comportements des agents dans le même environnement.

2. Structure du projet
Agent/
│
├─ venv/                     # environnement virtuel Python
├─ gridworld.py              # définition de l'environnement GridWorld
├─ agents/
│   ├─ policy_iteration.py
│   ├─ value_iteration.py
│   ├─ monte_carlo.py
│   └─ q_learning.py
├─ test_agents.py            # script pour entraîner et tester les agents
└─ README.md                 # ce fichier

3. Installation et setup

Créer un environnement virtuel Python :

python -m venv venv


Activer l’environnement :

Windows (PowerShell) :

.\venv\Scripts\Activate.ps1


Windows (cmd) :

.\venv\Scripts\activate.bat


Linux / MacOS :

source venv/bin/activate


Installer les dépendances :

pip install gymnasium numpy matplotlib

4. Exécution du projet

Assurez-vous que l’environnement est activé (venv).

Lancer le script de test des agents :

python test_agents.py


Chaque agent s’entraîne sur le GridWorld et affiche sa politique finale dans la console.

L’agent se déplace ensuite sur la grille avec rendu graphique Matplotlib.

Le script passe successivement par les quatre agents pour que vous puissiez observer et comparer leurs comportements.