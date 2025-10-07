from gridworld import MyGridWorld
from agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import csv

# --- Paramètres ---
sizes = [3, 4, 5, 6, 7, 8, 9, 10]
episodes_per_size = 50
start = (0, 0)

# === PARAMÈTRES DYNAMIQUES À TESTER ===
# Densité d'obstacles (pourcentage de la grille)
OBSTACLE_DENSITY = 0.15  # 15% de la grille
# Nombre de goals
NUM_GOALS = 1  # 1, 2, 3, ou 4
# Seed pour reproductibilité
RANDOM_SEED = 42

# --- Fonctions de génération dynamique ---
def generate_dynamic_obstacles(size, density=0.15, seed=None):
    """
    Génère des obstacles proportionnels à la taille de la grille.
    
    Args:
        size: Taille de la grille (size x size)
        density: Pourcentage de cellules avec obstacles (0.15 = 15%)
        seed: Graine aléatoire pour reproductibilité
    
    Returns:
        Liste de tuples (x, y) des obstacles
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_obstacles = max(1, int(size * size * density))
    obstacles = []
    max_attempts = size * size * 2
    attempts = 0
    
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        
        # Éviter départ et goal principal
        if (x, y) != (0, 0) and (x, y) != (size-1, size-1):
            if (x, y) not in obstacles:
                obstacles.append((x, y))
        attempts += 1
    
    return obstacles

def generate_dynamic_goals(size, num_goals=1):
    """
    Génère des goals en fonction de la taille de la grille.
    
    Args:
        size: Taille de la grille
        num_goals: Nombre de goals (1 à 4)
    
    Returns:
        Liste de tuples (x, y) des goals
    """
    if num_goals == 1:
        return [(size-1, size-1)]
    
    # Positions stratégiques pour plusieurs goals
    all_positions = [
        (size-1, size-1),      # Coin bas-droite
        (0, size-1),            # Coin haut-droite
        (size-1, 0),            # Coin bas-gauche
        (size//2, size-1),      # Milieu-droite
    ]
    
    return all_positions[:min(num_goals, len(all_positions))]

# --- Stockage des résultats ---
all_cumulative_rewards = []
mean_rewards = []
std_rewards = []
num_obstacles_per_size = []
image_files = []

# --- Création du dossier ---
os.makedirs("figures_dynamic", exist_ok=True)

print(f"\n{'='*60}")
print(f"CONFIGURATION:")
print(f"  - Densité d'obstacles: {OBSTACLE_DENSITY*100}%")
print(f"  - Nombre de goals: {NUM_GOALS}")
print(f"  - Seed aléatoire: {RANDOM_SEED}")
print(f"{'='*60}")

# --- Entraînement pour différentes tailles ---
for idx, size in enumerate(sizes):
    print(f"\n=== Grille {size}x{size} ({idx+1}/{len(sizes)}) ===")
    
    # Génération dynamique
    goals = generate_dynamic_goals(size, num_goals=NUM_GOALS)
    obstacles = generate_dynamic_obstacles(size, density=OBSTACLE_DENSITY, seed=RANDOM_SEED + idx)
    
    num_obstacles_per_size.append(len(obstacles))
    
    print(f"  Goals ({len(goals)}): {goals}")
    print(f"  Obstacles ({len(obstacles)}): {obstacles[:3]}{'...' if len(obstacles) > 3 else ''}")
    
    # Création de l'environnement
    env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)
    
    # Initialisation de l'agent
    q_agent = QLearningAgent(env)
    
    # Stockage des récompenses
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
            avg_last_10 = np.mean(cumulative_rewards[-10:])
            print(f"  Épisode {ep+1}/{episodes_per_size} - Récompense: {total_reward:.2f} (Moy 10: {avg_last_10:.2f})")

    # Sauvegarde des résultats
    all_cumulative_rewards.append(cumulative_rewards)
    mean_rewards.append(np.mean(cumulative_rewards[-10:]))
    std_rewards.append(np.std(cumulative_rewards[-10:]))

    # --- Graphique individuel ---
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards, marker='o', markersize=4, alpha=0.7, label=f'{size}x{size}')
    plt.axhline(y=mean_rewards[-1], color='r', linestyle='--', 
                label=f'Moyenne finale: {mean_rewards[-1]:.2f}')
    plt.title(f'Grille {size}x{size} - {len(obstacles)} obstacles, {len(goals)} goal(s)', 
              fontsize=13, fontweight='bold')
    plt.xlabel('Épisode', fontsize=11)
    plt.ylabel('Retour cumulé', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    image_file = f"figures_dynamic/grid_{size}x{size}.png"
    plt.savefig(image_file, dpi=150, bbox_inches='tight')
    plt.close()
    image_files.append(image_file)
    
    env.close()

# --- Sauvegarde CSV ---
csv_file = "cumulative_rewards_dynamic.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Taille", "Obstacles", "Goals"] + [f"Ep{i+1}" for i in range(episodes_per_size)])
    
    for i, size in enumerate(sizes):
        row = [f"{size}x{size}", num_obstacles_per_size[i], NUM_GOALS]
        row.extend(all_cumulative_rewards[i])
        writer.writerow(row)

print(f"\n✅ Résultats sauvegardés dans '{csv_file}'")

# --- Génération du GIF ---
images = [imageio.imread(img) for img in image_files]
imageio.mimsave("performance_dynamic.gif", images, duration=1.5)
print("✅ GIF 'performance_dynamic.gif' créé!")

# --- FIGURE GLOBALE COMPLÈTE ---
fig = plt.figure(figsize=(20, 12))

# 1. Évolution des courbes d'apprentissage
ax1 = plt.subplot(2, 3, 1)
for i, size in enumerate(sizes):
    ax1.plot(all_cumulative_rewards[i], label=f'{size}x{size}', alpha=0.8, linewidth=1.5)
ax1.set_title('Évolution du retour cumulé par épisode', fontsize=13, fontweight='bold')
ax1.set_xlabel('Épisode', fontsize=11)
ax1.set_ylabel('Retour cumulé', fontsize=11)
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Performance moyenne vs taille
ax2 = plt.subplot(2, 3, 2)
ax2.errorbar(sizes, mean_rewards, yerr=std_rewards, fmt='o-', 
             color='red', markersize=8, capsize=5, linewidth=2)
ax2.set_title('Performance moyenne vs Taille de grille', fontsize=13, fontweight='bold')
ax2.set_xlabel('Taille de la grille', fontsize=11)
ax2.set_ylabel('Retour moyen ± écart-type', fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Nombre d'obstacles vs taille
ax3 = plt.subplot(2, 3, 3)
ax3.plot(sizes, num_obstacles_per_size, 'o-', color='orange', markersize=8, linewidth=2)
ax3.set_title(f'Nombre d\'obstacles (densité: {OBSTACLE_DENSITY*100}%)', 
              fontsize=13, fontweight='bold')
ax3.set_xlabel('Taille de la grille', fontsize=11)
ax3.set_ylabel('Nombre d\'obstacles', fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Courbes normalisées
ax4 = plt.subplot(2, 3, 4)
for i, size in enumerate(sizes):
    rewards = np.array(all_cumulative_rewards[i])
    if np.max(rewards) != np.min(rewards):
        normalized = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    else:
        normalized = np.zeros_like(rewards)
    ax4.plot(normalized, label=f'{size}x{size}', alpha=0.8, linewidth=1.5)
ax4.set_title('Courbes normalisées (tendances)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Épisode', fontsize=11)
ax4.set_ylabel('Retour normalisé [0-1]', fontsize=11)
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Bar chart des performances finales
ax5 = plt.subplot(2, 3, 5)
bars = ax5.bar(sizes, mean_rewards, color='skyblue', alpha=0.7, edgecolor='black')
ax5.errorbar(sizes, mean_rewards, yerr=std_rewards, fmt='none', 
             color='black', capsize=3, linewidth=1.5)
ax5.set_title('Performance finale par taille', fontsize=13, fontweight='bold')
ax5.set_xlabel('Taille de la grille', fontsize=11)
ax5.set_ylabel('Retour moyen (10 derniers épisodes)', fontsize=11)
ax5.grid(True, axis='y', alpha=0.3)

# Ajouter valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars, mean_rewards)):
    ax5.text(bar.get_x() + bar.get_width()/2, val + std_rewards[i], 
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# 6. Analyse de convergence (moyenne mobile)
ax6 = plt.subplot(2, 3, 6)
window = 5
for i, size in enumerate(sizes):
    rewards = np.array(all_cumulative_rewards[i])
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax6.plot(range(window-1, len(rewards)), moving_avg, 
             label=f'{size}x{size}', alpha=0.8, linewidth=1.5)
ax6.set_title(f'Moyenne mobile (fenêtre={window})', fontsize=13, fontweight='bold')
ax6.set_xlabel('Épisode', fontsize=11)
ax6.set_ylabel('Retour cumulé (lissé)', fontsize=11)
ax6.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax6.grid(True, alpha=0.3)

# Titre global
fig.suptitle(f'Analyse Q-Learning - Obstacles dynamiques ({OBSTACLE_DENSITY*100}%) - {NUM_GOALS} goal(s)', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig("analyse_complete_dynamic.png", dpi=300, bbox_inches='tight')
print("✅ Figure complète sauvegardée: 'analyse_complete_dynamic.png'")
plt.show(block=True)

# --- Résultats finaux ---
print(f"\n{'='*60}")
print("RÉSULTATS FINAUX:")
print(f"{'='*60}")
print(f"{'Taille':<10} {'Obstacles':<12} {'Retour Moyen':<15} {'Écart-type':<12}")
print(f"{'-'*60}")
for i, size in enumerate(sizes):
    print(f"{size}x{size:<7} {num_obstacles_per_size[i]:<12} "
          f"{mean_rewards[i]:<15.2f} {std_rewards[i]:<12.2f}")
print(f"{'='*60}")

print(f"\n✅ Figures individuelles: dossier 'figures_dynamic/'")
print(f"✅ Analyse complète: 'analyse_complete_dynamic.png'")
print(f"✅ Données CSV: '{csv_file}'")
print(f"✅ Animation GIF: 'performance_dynamic.gif'")