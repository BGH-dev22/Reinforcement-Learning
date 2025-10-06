from gridworld import MyGridWorld
from agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import csv

# --- Paramètres ---
sizes = [3, 4, 5, 6, 7, 8, 9, 10]  # Tailles de grille à tester
episodes_per_size = 50  # Nombre d'épisodes par taille
start = (0, 0)
obstacles = [(1, 1), (2, 3)]

# --- Stockage des résultats ---
all_cumulative_rewards = []  # Pour chaque taille: liste des retours cumulés par épisode
mean_rewards = []  # Retour moyen par taille
image_files = []  # Pour stocker les fichiers images qui formeront le GIF

# --- Création du dossier pour sauvegarder les images ---
os.makedirs("figures", exist_ok=True)

# --- Entraînement pour différentes tailles de grille ---
for size in sizes:
    print(f"\n=== Entraînement pour grille {size}x{size} ===")
    
    # Création de l'environnement adapté à la taille
    goals = [(size-1, size-1)]  # Goal unique dans le coin opposé
    env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)
    
    # Initialisation de l'agent
    q_agent = QLearningAgent(env)
    
    # Stockage des récompenses pour cette taille
    cumulative_rewards = []
    
    # Entraînement
    for ep in range(episodes_per_size):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = q_agent.choose_action(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            q_agent.update(tuple(state), action, reward, tuple(next_state))
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        cumulative_rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            print(f"Épisode {ep+1}/{episodes_per_size} - Récompense: {total_reward:.2f}")

    # Sauvegarde des résultats
    all_cumulative_rewards.append(cumulative_rewards)
    mean_rewards.append(np.mean(cumulative_rewards[-10:]))  # Moyenne des 10 derniers épisodes

    # --- Sauvegarde du graphique pour le GIF ---
    plt.figure(figsize=(8,5))
    plt.plot(cumulative_rewards, marker='o', label=f'{size}x{size}')
    plt.title(f'Retour cumulé par épisode - Grille {size}x{size}')
    plt.xlabel('Épisode')
    plt.ylabel('Retour cumulé')
    plt.grid(True)
    plt.legend()
    image_file = f"figures/grid_{size}x{size}.png"
    plt.savefig(image_file)
    plt.close()
    image_files.append(image_file)
    
    env.close()

# --- Sauvegarder les retours cumulés dans un fichier CSV lisible ---
csv_file = "cumulative_rewards.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    for i, rewards in enumerate(all_cumulative_rewards):
        writer.writerow([f"{sizes[i]}x{sizes[i]}"] + list(rewards))

print(f"\n✅ Les retours cumulés sont enregistrés dans '{csv_file}'")

# --- Générer un GIF à partir des images ---
images = [imageio.imread(img) for img in image_files]
imageio.mimsave("performance_gridworld.gif", images, duration=1)  # 1 sec par image
print("✅ GIF 'performance_gridworld.gif' créé avec succès!")

# --- Création de la figure globale récapitulative ---
plt.figure(figsize=(18, 12))

# Graphique 1: Toutes les courbes d'apprentissage
plt.subplot(2, 2, 1)
for i, size in enumerate(sizes):
    plt.plot(all_cumulative_rewards[i], label=f'{size}x{size}')
plt.title('Évolution du retour cumulé par épisode', fontsize=14)
plt.xlabel('Épisode', fontsize=12)
plt.ylabel('Retour cumulé', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

# Graphique 2: Performance moyenne en fonction de la taille
plt.subplot(2, 2, 2)
plt.plot(sizes, mean_rewards, 'o-', color='red', markersize=8)
plt.title('Performance moyenne vs Taille de la grille', fontsize=14)
plt.xlabel('Taille de la grille', fontsize=12)
plt.ylabel('Retour moyen (10 derniers épisodes)', fontsize=12)
plt.grid(True)

# Graphique 3: Courbes normalisées
plt.subplot(2, 2, 3)
for i, size in enumerate(sizes):
    normalized_rewards = np.array(all_cumulative_rewards[i])
    if np.max(normalized_rewards) != np.min(normalized_rewards):
        normalized_rewards = (normalized_rewards - np.min(normalized_rewards)) / (np.max(normalized_rewards) - np.min(normalized_rewards))
    else:
        normalized_rewards = np.zeros_like(normalized_rewards)
    plt.plot(normalized_rewards, label=f'{size}x{size}')
plt.title('Courbes normalisées (comparaison des tendances)', fontsize=14)
plt.xlabel('Épisode', fontsize=12)
plt.ylabel('Retour normalisé', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

# Graphique 4: Comparaison des performances finales
plt.subplot(2, 2, 4)
plt.bar(sizes, mean_rewards, color='skyblue', alpha=0.7)
plt.title('Performance finale par taille de grille', fontsize=14)
plt.xlabel('Taille de la grille', fontsize=12)
plt.ylabel('Retour moyen (10 derniers épisodes)', fontsize=12)
plt.grid(True, axis='y')

plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.suptitle('Analyse complète de la performance de Q-Learning', fontsize=16, y=1.02)
plt.savefig("analyse_complete.png", dpi=300, bbox_inches='tight')
plt.show(block=True)

# --- Affichage des résultats finaux ---
print("\n=== Résultats finaux ===")
for i, size in enumerate(sizes):
    print(f"Grille {size}x{size}: Retour moyen = {mean_rewards[i]:.2f}")

print("✅ Les figures individuelles sont dans le dossier 'figures/'")
print("✅ L'analyse complète est sauvegardée dans 'analyse_complete.png'")
