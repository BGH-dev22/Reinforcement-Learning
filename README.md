# Q-Learning GridWorld - Analyse des Paramètres

## 🎯 Objectif du projet

Ce projet a pour but de tester et analyser l'influence des principaux paramètres du **Q-Learning** sur la performance d'un agent dans un environnement **GridWorld** simple.  

Les paramètres étudiés sont :
- **Learning Rate (α)** : vitesse d'apprentissage
- **Discount Factor (γ)** : importance des récompenses futures
- **Exploration Rate (ε)** : propension de l'agent à explorer
- **Exploration Decay** : décroissance de l'exploration au fil du temps
- **Exploration Min** : exploration minimale autorisée

L'objectif est de visualiser comment ces paramètres influencent la fonction de valeur, la politique optimale et la performance de l'agent.

---

## 📂 Structure du projet

.
├── agents/
│ ├── q_learning.py # Classe QLearningAgent et fonctions d'entraînement
│ ├── mc.py # (optionnel) Monte Carlo
│ ├── value_iteration.py # Itération de valeur
│ └── policy_iteration.py # Itération de politique
├── figures_qlearning/ # Graphiques générés pour analyse des paramètres
├── results/ # Résultats sauvegardés
├── gridworld.py # Définition de l'environnement GridWorld
├── test_agents.py # Script principal pour tester l'agent et analyser les paramètres
├── q_learning_agent.pkl # Q-table sauvegardée
├── q_learning_test_results.png # Visualisation finale
└── README.md # Ce fichier

yaml
Copier le code

---

## 📝 Contenu du script `test_agents.py`

1. **Initialisation de l'environnement GridWorld** :
   - Taille de la grille : 5x5
   - État initial : `(0,0)`
   - Objectif : `(4,4)`
   - Obstacles : `(1,1)` et `(2,3)`

2. **Création de l'agent Q-Learning** :
   - Paramètres configurables : `learning_rate`, `discount_factor`, `exploration_rate`, `exploration_decay`, `exploration_min`

3. **Entraînement de l'agent** :
   - Nombre d'épisodes : 100
   - Historique des récompenses et des pas effectués

4. **Analyse et visualisation** :
   - Courbe d'apprentissage (récompenses cumulées)
   - Nombre de pas par épisode
   - Fonction de valeur V(s) = max_a Q(s,a)
   - Politique optimale π(s) affichée avec des flèches
   - Sauvegarde des visualisations (`q_learning_test_results.png`)

5. **Test de la politique apprise** :
   - Exploration désactivée (`ε=0`)
   - Vérification des performances de l'agent sur plusieurs épisodes

6. **Sauvegarde et chargement de la Q-table** :
   - Sauvegarde dans `q_learning_agent.pkl`
   - Vérification que la Q-table chargée correspond à l'originale

---

## 📊 Visualisation

Le script produit plusieurs visualisations permettant de comprendre l'impact des paramètres sur :

- La vitesse et la stabilité de l'apprentissage
- La convergence vers la politique optimale
- La qualité des récompenses cumulées

---

## 🚀 Exécution

1. Créer un environnement virtuel et installer les dépendances :

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install numpy matplotlib
Exécuter le script principal :

bash
Copier le code
python test_agents.py
Les résultats seront affichés et sauvegardés sous :

q_learning_test_results.png

q_learning_agent.pkl

📌 Notes importantes
L'agent utilise un epsilon-greedy policy, avec un exploration_decay pour réduire l'exploration progressivement.

Les paramètres peuvent être facilement modifiés dans test_agents.py pour analyser leur influence.

La limite maximale de pas par épisode est fixée à 50 pour éviter que l'agent tourne indéfiniment dans la grille.

