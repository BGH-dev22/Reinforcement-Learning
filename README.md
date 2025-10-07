# 🧠 TD(0) Agent avec Approximation Linéaire — GridWorld

## 📘 Description du projet
Ce projet implémente et visualise **un agent TD(0) (Temporal Difference)** avec **approximation linéaire de la fonction de valeur** dans un environnement **GridWorld** personnalisé.  
L’objectif est de démontrer comment un agent apprend à estimer la valeur des états et à améliorer sa politique de déplacement par renforcement progressif à travers l’exploration et la mise à jour des poids linéaires.

---

## 🏗️ Structure du projet

📂 Project Root
│
├── agents/ # Contient les classes des agents (ex: LinearTDAgent)
│ ├── init.py
│ └── agent_td.py
│
├── figures_td0/ # Dossier contenant les visualisations générées
│ ├── learning_analysis.png
│ ├── policy_visualization.png
│ └── weights_analysis.png
│
├── results_deep_qlearning/ # (optionnel) Dossier pour d’autres expérimentations
│
├── gridworld.py # Environnement GridWorld personnalisé
├── test_agent_td.py # Script principal d’entraînement et de visualisation TD(0)
├── test_agents.py # (optionnel) Autres tests ou comparaisons d’agents
│
├── td0_rewards.npy # Récompenses enregistrées (format binaire)
├── td0_rewards.txt # Récompenses enregistrées (format texte)
├── td0_weights.npy # Poids appris de l’agent
│
├── venv/ # Environnement virtuel Python
└── README.md # Ce fichier

yaml
Copier le code

---

## ⚙️ Installation

### 1️⃣ Créer et activer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate       # sous Linux/Mac
venv\Scripts\activate          # sous Windows
2️⃣ Installer les dépendances requises
bash
Copier le code
pip install numpy matplotlib seaborn
🚀 Exécution
Pour lancer l’expérience TD(0), exécute simplement :

bash
Copier le code
python test_agent_td.py
🔹 Ce que le script fait :
Initialise un environnement MyGridWorld avec une taille de 6x6, un point de départ, un but et des obstacles.

Crée un agent TD(0) avec approximation linéaire.

Entraîne l’agent pendant 300 épisodes.

Génère automatiquement plusieurs visualisations :

Courbe d’apprentissage des récompenses

Évolution du taux d’exploration

Heatmap des poids appris

Politique optimale apprise

Trajectoire de l’agent dans la grille

Sauvegarde les résultats (récompenses, poids, figures).

📊 Sorties générées
Fichier / Dossier	Description
figures_td0/learning_analysis.png	Courbes d’évolution des récompenses, pas et exploration
figures_td0/weights_analysis.png	Analyse détaillée des poids appris
figures_td0/policy_visualization.png	Politique finale et trajectoire optimale
td0_rewards.npy	Historique complet des récompenses (format NumPy)
td0_rewards.txt	Résumé textuel avec statistiques
td0_weights.npy	Poids finaux de l’agent (format NumPy)

🧩 Détails de l’implémentation
🔸 Agent TD(0)
L’agent suit la mise à jour suivante :

𝛿
=
𝑟
+
𝛾
𝑉
(
𝑠
′
)
−
𝑉
(
𝑠
)
δ=r+γV(s 
′
 )−V(s)
𝑤
←
𝑤
+
𝛼
 
𝛿
 
𝜙
(
𝑠
)
w←w+αδϕ(s)
Fonction d’approximation : linéaire avec features
[x, y, x², y², x·y, prox_goal, bias]

Politique : ε-greedy (avec décroissance exponentielle de ε)

Apprentissage : mise à jour incrémentale à chaque étape

🔸 Environnement GridWorld
Taille : 6x6

Départ : (0,0)

But : (5,5)

Obstacles : (1,1), (2,3), (3,3)

Récompenses :

Positive à l’arrivée

Négative pour les obstacles ou les sorties de grille

📈 Exemple de sortie console
yaml
Copier le code
================================================================================
🎯 DÉMONSTRATION: Agent TD(0) avec Approximation Linéaire
================================================================================

🏃 Entraînement sur 300 épisodes...
  Episode 50/300 - Reward: -2.10, Steps: 40.2, ε: 0.61
  Episode 100/300 - Reward: -1.50, Steps: 28.4, ε: 0.37
  ...
✅ Entraînement terminé!
   Performance finale: 0.85
Les résultats sont automatiquement enregistrés et affichés dans figures_td0/.

🎓 Objectif pédagogique
Ce projet a pour but de :

Illustrer le fonctionnement de TD(0) dans un environnement discret.

Montrer comment une fonction de valeur peut être apprise par approximation linéaire.

Comprendre l’impact du taux d’apprentissage, du facteur de discount et de l’exploration sur la convergence.

Fournir des visualisations claires pour interpréter les résultats.
