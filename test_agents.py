from gridworld import MyGridWorld
from agents.q_learning import QLearningAgent, train_q_learning_agent
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("🎯 TEST SIMPLE: Agent Q-Learning")
print("="*70)

# Créer l'environnement
size = 5
start = (0, 0)
goals = [(4, 4)]
obstacles = [(1, 1), (2, 3)]

env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)

print(f"\n📋 Configuration de l'environnement:")
print(f"   - Taille: {size}x{size}")
print(f"   - Départ: {start}")
print(f"   - Objectif: {goals}")
print(f"   - Obstacles: {obstacles}")

# Créer l'agent Q-Learning
print("\n" + "="*70)
agent = QLearningAgent(
    env,
    learning_rate=0.1,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.995,
    exploration_min=0.01
)

# Entraînement
print("\n" + "="*70)
print("🏃 ENTRAÎNEMENT")
print("="*70)

episodes = 100
rewards_history, steps_history = train_q_learning_agent(env, agent, episodes, verbose=True)

print("\n✅ Entraînement terminé!")
print(f"   Performance finale (10 derniers): {np.mean(rewards_history[-10:]):.2f}")
print(f"   Pas moyens (10 derniers): {np.mean(steps_history[-10:]):.1f}")

# Statistiques de l'agent
agent.print_statistics()

# Test de la politique apprise
print("\n" + "="*70)
print("🎮 TEST DE LA POLITIQUE APPRISE")
print("="*70)

# Désactiver l'exploration pour tester
agent.exploration_rate = 0.0

test_episodes = 5
print(f"\nTest sur {test_episodes} épisodes (ε=0, greedy policy):\n")

for ep in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    path = [tuple(state)]
    
    while not done and steps < 50:  # Limite de sécurité
        action = agent.choose_action(tuple(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        state = next_state
        total_reward += reward
        steps += 1
        path.append(tuple(state))
        done = terminated or truncated
    
    print(f"  Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}, "
          f"Success={'✅' if done and reward > 0 else '❌'}")
    print(f"    Chemin: {' → '.join([str(p) for p in path[:10]])}" + 
          ("..." if len(path) > 10 else ""))

# Visualisation
print("\n" + "="*70)
print("📊 VISUALISATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Courbe d'apprentissage
ax1 = axes[0, 0]
ax1.plot(rewards_history, alpha=0.5, linewidth=0.8)
window = 10
smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
ax1.plot(range(len(smoothed)), smoothed, 'r-', linewidth=2, label=f'Moyenne mobile ({window})')
ax1.set_title('Courbe d\'apprentissage', fontsize=13, fontweight='bold')
ax1.set_xlabel('Épisode')
ax1.set_ylabel('Retour cumulé')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Nombre de pas
ax2 = axes[0, 1]
ax2.plot(steps_history, alpha=0.5, linewidth=0.8, color='green')
smoothed_steps = np.convolve(steps_history, np.ones(window)/window, mode='valid')
ax2.plot(range(len(smoothed_steps)), smoothed_steps, 'darkgreen', linewidth=2)
ax2.set_title('Nombre de pas par épisode', fontsize=13, fontweight='bold')
ax2.set_xlabel('Épisode')
ax2.set_ylabel('Nombre de pas')
ax2.grid(True, alpha=0.3)

# 3. Grille des valeurs V(s)
ax3 = axes[1, 0]
value_grid = agent.get_value_grid()
im = ax3.imshow(value_grid, cmap='viridis', origin='upper')
ax3.set_title('Fonction de valeur V(s) = max_a Q(s,a)', fontsize=13, fontweight='bold')
ax3.set_xlabel('y')
ax3.set_ylabel('x')

# Marquer start, goal, obstacles
ax3.plot(start[1], start[0], 'g*', markersize=20, label='Start')
for goal in goals:
    ax3.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
for obs in obstacles:
    ax3.plot(obs[1], obs[0], 'kx', markersize=15, markeredgewidth=3)

plt.colorbar(im, ax=ax3, label='Valeur')

# 4. Politique optimale (flèches)
ax4 = axes[1, 1]
ax4.imshow(value_grid, cmap='viridis', origin='upper', alpha=0.3)

# Directions des flèches: 0=haut, 1=bas, 2=gauche, 3=droite
arrow_map = {
    0: (0, -0.3),   # Haut
    1: (0, 0.3),    # Bas
    2: (-0.3, 0),   # Gauche
    3: (0.3, 0)     # Droite
}

policy_grid = agent.get_policy_grid()

for x in range(size):
    for y in range(size):
        if (x, y) not in obstacles and (x, y) not in goals:
            action = policy_grid[x, y]
            dx, dy = arrow_map[action]
            ax4.arrow(y, x, dy, dx, head_width=0.2, head_length=0.15,
                     fc='white', ec='black', linewidth=1.5)

ax4.set_title('Politique optimale π(s)', fontsize=13, fontweight='bold')
ax4.set_xlabel('y')
ax4.set_ylabel('x')
ax4.plot(start[1], start[0], 'g*', markersize=20)
for goal in goals:
    ax4.plot(goal[1], goal[0], 'r*', markersize=20)
for obs in obstacles:
    ax4.plot(obs[1], obs[0], 'kx', markersize=15, markeredgewidth=3)

plt.suptitle('Analyse de l\'agent Q-Learning', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('q_learning_test_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualisation sauvegardée dans 'q_learning_test_results.png'")

# Sauvegarder la Q-table
agent.save_q_table('q_learning_agent.pkl')

# Test de chargement
print("\n🔄 Test de sauvegarde/chargement...")
agent2 = QLearningAgent(env, learning_rate=0.1)
agent2.load_q_table('q_learning_agent.pkl')

# Vérifier que les Q-values sont identiques
test_state = (1, 0)
print(f"\nVérification: Q-values pour l'état {test_state}")
print(f"  Agent original: {agent.get_all_q_values(test_state)}")
print(f"  Agent chargé:   {agent2.get_all_q_values(test_state)}")

env.close()

print("\n" + "="*70)
print("✅ TEST TERMINÉ AVEC SUCCÈS!")
print("="*70)