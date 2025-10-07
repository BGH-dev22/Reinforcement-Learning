from gridworld import MyGridWorld
from agents.agent_td import LinearTDAgent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
sns.set_style("whitegrid")
os.makedirs("figures_td0", exist_ok=True)

print("="*80)
print("🎯 DÉMONSTRATION: Agent TD(0) avec Approximation Linéaire")
print("="*80)
print("\nBasé sur le pseudo-code:")
print("  φ(s): → vecteur R^d ; V(s,w)=w^T φ(s)")
print("  δ = r + (0 if done else γ·w^T·φ(s2)) - w^T·φ(s)")
print("  w ← w + α · δ · φ(s)")
print("="*80)

# --- Paramètres ---
size = 6
start = (0, 0)
goals = [(5, 5)]
obstacles = [(1, 1), (2, 3), (3, 3)]
episodes = 300

# =============================================================================
# PARTIE 1: Entraînement de l'agent TD(0)
# =============================================================================
print("\n" + "="*80)
print("PARTIE 1: Entraînement de l'agent TD(0)")
print("="*80)

env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)

# Créer l'agent
agent = LinearTDAgent(
    env,
    learning_rate=0.01,  # α
    discount_factor=0.99,  # γ
    exploration_rate=1.0,
    exploration_decay=0.995,
    exploration_min=0.01
)

# Entraînement
print(f"\n🏃 Entraînement sur {episodes} épisodes...")
rewards_history = []
steps_history = []
exploration_history = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = agent.choose_action(tuple(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Mise à jour TD(0)
        agent.update(tuple(state), action, reward, tuple(next_state), 
                    done=(terminated or truncated))
        
        state = next_state
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    rewards_history.append(total_reward)
    steps_history.append(steps)
    exploration_history.append(agent.exploration_rate)
    
    if (ep + 1) % 50 == 0:
        avg_reward = np.mean(rewards_history[-20:])
        avg_steps = np.mean(steps_history[-20:])
        print(f"  Episode {ep+1}/{episodes} - Reward: {avg_reward:.2f}, Steps: {avg_steps:.1f}, ε: {agent.exploration_rate:.3f}")

print(f"\n✅ Entraînement terminé!")
print(f"   Performance finale: {np.mean(rewards_history[-20:]):.2f}")

# =============================================================================
# PARTIE 2: Visualisation de l'apprentissage
# =============================================================================
print("\n" + "="*80)
print("PARTIE 2: Visualisation de l'apprentissage")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Courbe d'apprentissage (récompenses)
ax1 = axes[0, 0]
ax1.plot(rewards_history, alpha=0.6, linewidth=0.8, label='Récompense brute')
# Moyenne mobile
window = 20
smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
ax1.plot(range(len(smoothed)), smoothed, 'r-', linewidth=2, label=f'Moyenne mobile ({window})')
ax1.set_title('Courbe d\'apprentissage - Récompenses', fontsize=13, fontweight='bold')
ax1.set_xlabel('Épisode')
ax1.set_ylabel('Retour cumulé')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Nombre de pas par épisode
ax2 = axes[0, 1]
ax2.plot(steps_history, alpha=0.6, linewidth=0.8, color='green')
smoothed_steps = np.convolve(steps_history, np.ones(window)/window, mode='valid')
ax2.plot(range(len(smoothed_steps)), smoothed_steps, 'darkgreen', linewidth=2)
ax2.set_title('Efficacité - Nombre de pas', fontsize=13, fontweight='bold')
ax2.set_xlabel('Épisode')
ax2.set_ylabel('Nombre de pas')
ax2.grid(True, alpha=0.3)

# Subplot 3: Décroissance de l'exploration
ax3 = axes[1, 0]
ax3.plot(exploration_history, color='orange', linewidth=2)
ax3.set_title('Évolution du taux d\'exploration (ε)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Épisode')
ax3.set_ylabel('ε (exploration rate)')
ax3.axhline(y=agent.exploration_min, color='r', linestyle='--', label='ε_min')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Distribution des récompenses finales
ax4 = axes[1, 1]
final_rewards = rewards_history[-50:]
ax4.hist(final_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax4.axvline(np.mean(final_rewards), color='r', linestyle='--', linewidth=2, label=f'Moyenne: {np.mean(final_rewards):.2f}')
ax4.set_title('Distribution des performances finales (50 derniers épisodes)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Retour cumulé')
ax4.set_ylabel('Fréquence')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Analyse de l\'agent TD(0) avec Approximation Linéaire', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('figures_td0/learning_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# PARTIE 3: Analyse des poids et features
# =============================================================================
print("\n" + "="*80)
print("PARTIE 3: Analyse des poids appris")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Poids par action et par feature
ax1 = axes[0, 0]
feature_names = ['x', 'y', 'x²', 'y²', 'x·y', 'prox_goal', 'bias']
action_names = ['↑ Haut', '↓ Bas', '← Gauche', '→ Droite']

weights_matrix = agent.weights.reshape(agent.n_actions, agent.n_features)
x_pos = np.arange(len(feature_names))
width = 0.18

for i, action_name in enumerate(action_names):
    ax1.bar(x_pos + i*width, weights_matrix[i], width, label=action_name, alpha=0.8)

ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(feature_names, rotation=30, ha='right')
ax1.set_title('Poids w appris par feature et par action', fontsize=13, fontweight='bold')
ax1.set_ylabel('Valeur du poids')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linewidth=0.8)

# Subplot 2: Heatmap des poids
ax2 = axes[0, 1]
im = ax2.imshow(weights_matrix, cmap='coolwarm', aspect='auto', vmin=-np.max(np.abs(weights_matrix)), vmax=np.max(np.abs(weights_matrix)))
ax2.set_xticks(range(len(feature_names)))
ax2.set_xticklabels(feature_names, rotation=45, ha='right')
ax2.set_yticks(range(len(action_names)))
ax2.set_yticklabels(action_names)
ax2.set_title('Heatmap de la matrice des poids', fontsize=13, fontweight='bold')

# Ajouter les valeurs dans chaque cellule
for i in range(len(action_names)):
    for j in range(len(feature_names)):
        text = ax2.text(j, i, f'{weights_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax2, label='Valeur')

# Subplot 3: Norme des poids par action
ax3 = axes[1, 0]
weight_norms = [np.linalg.norm(weights_matrix[i]) for i in range(agent.n_actions)]
colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
bars = ax3.bar(action_names, weight_norms, color=colors, alpha=0.7, edgecolor='black')
ax3.set_title('Norme L2 des poids par action', fontsize=13, fontweight='bold')
ax3.set_ylabel('||w_action||₂')
ax3.grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bar, norm in zip(bars, weight_norms):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{norm:.2f}', ha='center', va='bottom', fontweight='bold')

# Subplot 4: Statistiques des poids
ax4 = axes[1, 1]
ax4.axis('off')

stats_text = "📊 STATISTIQUES DES POIDS\n" + "="*40 + "\n\n"
stats_text += f"Dimension totale: {len(agent.weights)}\n"
stats_text += f"  • Par action: {agent.n_features} features\n"
stats_text += f"  • Nombre d'actions: {agent.n_actions}\n\n"
stats_text += f"Poids global:\n"
stats_text += f"  • Maximum: {np.max(agent.weights):.4f}\n"
stats_text += f"  • Minimum: {np.min(agent.weights):.4f}\n"
stats_text += f"  • Moyenne: {np.mean(agent.weights):.4f}\n"
stats_text += f"  • Écart-type: {np.std(agent.weights):.4f}\n"
stats_text += f"  • Norme L2: {np.linalg.norm(agent.weights):.4f}\n\n"
stats_text += "Poids par action:\n"

for i, action in enumerate(action_names):
    stats_text += f"  {action}:\n"
    stats_text += f"    • Max: {np.max(weights_matrix[i]):.4f}\n"
    stats_text += f"    • Min: {np.min(weights_matrix[i]):.4f}\n"
    stats_text += f"    • Moyenne: {np.mean(weights_matrix[i]):.4f}\n"
    stats_text += f"    • Norme L2: {np.linalg.norm(weights_matrix[i]):.4f}\n"

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Analyse des poids appris par l\'agent TD(0)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('figures_td0/weights_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# PARTIE 4: Visualisation de la politique apprise
# =============================================================================
print("\n" + "="*80)
print("PARTIE 4: Visualisation de la politique apprise")
print("="*80)

# Obtenir la grille de valeurs
value_grid = agent.get_value_grid()

# Créer une figure pour la politique et les valeurs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Subplot 1: Carte des valeurs
im = ax1.imshow(value_grid.T, cmap='viridis', origin='lower', interpolation='bilinear')
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Valeur d\'état V(s)', fontsize=12)

# Ajouter les obstacles
for obs in obstacles:
    ax1.add_patch(plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, fill=True, color='red', alpha=0.7))

# Ajouter le départ et le goal
ax1.add_patch(plt.Circle(start, 0.3, color='green', alpha=0.7))
ax1.add_patch(plt.Circle(goals[0], 0.3, color='gold', alpha=0.7))

# Ajouter les flèches pour la politique
action_symbols = ['↑', '↓', '←', '→']
for x in range(size):
    for y in range(size):
        if (x, y) not in obstacles:
            action = agent.get_policy((x, y))
            ax1.text(x, y, action_symbols[action], ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')

ax1.set_title('Politique apprise et valeurs d\'état', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_xticks(range(size))
ax1.set_yticks(range(size))
ax1.grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Trajectoire optimale
ax2.imshow(value_grid.T, cmap='viridis', origin='lower', alpha=0.3)

# Ajouter les obstacles
for obs in obstacles:
    ax2.add_patch(plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, fill=True, color='red', alpha=0.7))

# Ajouter le départ et le goal
ax2.add_patch(plt.Circle(start, 0.3, color='green', alpha=0.7))
ax2.add_patch(plt.Circle(goals[0], 0.3, color='gold', alpha=0.7))

# Simuler une trajectoire avec la politique apprise
state, _ = env.reset()
trajectory = [tuple(state)]
done = False
max_steps = 50

while not done and len(trajectory) < max_steps:
    action = agent.get_policy(tuple(state))
    next_state, _, terminated, truncated, _ = env.step(action)
    state = next_state
    trajectory.append(tuple(state))
    done = terminated or truncated

# Dessiner la trajectoire
for i in range(len(trajectory) - 1):
    x1, y1 = trajectory[i]
    x2, y2 = trajectory[i + 1]
    ax2.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.2, head_length=0.2, 
             fc='blue', ec='blue', alpha=0.7, linewidth=2)

ax2.set_title('Trajectoire optimale avec la politique apprise', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_xticks(range(size))
ax2.set_yticks(range(size))
ax2.grid(True, linestyle='--', alpha=0.7)

plt.suptitle('Visualisation de la politique apprise', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures_td0/policy_visualization.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# PARTIE 5: Sauvegarde des résultats
# =============================================================================
print("\n" + "="*80)
print("PARTIE 5: Sauvegarde des résultats")
print("="*80)

# Sauvegarder les récompenses dans un fichier .npy et .txt
try:
    # Sauvegarde en format binaire .npy
    np.save("td0_rewards.npy", rewards_history)
    
    # Sauvegarde en format texte .txt pour vérification
    with open("td0_rewards.txt", "w") as f:
        f.write("Retours cumulés par épisode pour l'agent TD(0)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Paramètres:\n")
        f.write(f"  - Taille de la grille: {size}x{size}\n")
        f.write(f"  - Nombre d'épisodes: {episodes}\n")
        f.write(f"  - Taux d'apprentissage: {agent.learning_rate}\n")
        f.write(f"  - Facteur de discount: {agent.discount_factor}\n\n")
        
        f.write("Récompenses par épisode:\n")
        for i, reward in enumerate(rewards_history):
            f.write(f"  Épisode {i+1}: {reward:.4f}\n")
        
        f.write(f"\nStatistiques finales:\n")
        f.write(f"  - Moyenne (tous épisodes): {np.mean(rewards_history):.4f}\n")
        f.write(f"  - Moyenne (20 derniers): {np.mean(rewards_history[-20:]):.4f}\n")
        f.write(f"  - Maximum: {np.max(rewards_history):.4f}\n")
        f.write(f"  - Minimum: {np.min(rewards_history):.4f}\n")
        f.write(f"  - Écart-type: {np.std(rewards_history):.4f}\n")
    
    print("✅ Fichiers de récompenses créés avec succès!")
    print("  - td0_rewards.npy (format binaire)")
    print("  - td0_rewards.txt (format texte)")
    
    # Sauvegarder les poids de l'agent
    agent.save_weights("td0_weights.npy")
    
    # Vérification que les fichiers existent
    if os.path.exists("td0_rewards.npy") and os.path.exists("td0_rewards.txt") and os.path.exists("td0_weights.npy"):
        print("✅ Tous les fichiers ont été créés avec succès")
    else:
        print("❌ Erreur: Certains fichiers n'ont pas été créés correctement")
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde des fichiers: {str(e)}")

# Afficher les résultats finaux
print("\n" + "="*80)
print("RÉSULTATS FINAUX")
print("="*80)
print(f"Performance moyenne (20 derniers épisodes): {np.mean(rewards_history[-20:]):.4f}")
print(f"Nombre de pas moyen (20 derniers épisodes): {np.mean(steps_history[-20:]):.2f}")
print(f"Taux d'exploration final: {agent.exploration_rate:.4f}")
print(f"Norme L2 des poids: {np.linalg.norm(agent.weights):.4f}")

print("\n✅ Tous les résultats ont été sauvegardés dans:")
print("  - figures_td0/ (dossier avec les visualisations)")
print("  - td0_rewards.npy et td0_rewards.txt (récompenses)")
print("  - td0_weights.npy (poids de l'agent)")