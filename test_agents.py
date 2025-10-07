from gridworld import MyGridWorld
from agents.q_learning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import csv
from itertools import product

# --- Paramètres de l'environnement (fixes) ---
size = 5
start = (0, 0)
goals = [(4, 4)]
obstacles = [(1, 1), (2, 3)]
episodes = 100

# --- Hyperparamètres à tester ---
learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
discount_factors = [0.8, 0.9, 0.95, 0.99]
exploration_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
exploration_decays = [0.99, 0.995, 0.999, 1.0]  # 1.0 = pas de décroissance

# --- Création des dossiers ---
os.makedirs("figures_qlearning", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =============================================================================
# TEST 1: Influence du Learning Rate (α)
# =============================================================================
print("\n" + "="*60)
print("TEST 1: Influence du Learning Rate (α)")
print("="*60)

results_lr = []
for lr in learning_rates:
    print(f"\n→ Test avec α = {lr}")
    env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)
    agent = QLearningAgent(env, learning_rate=lr)
    
    cumulative_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(tuple(state), action, reward, tuple(next_state))
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        cumulative_rewards.append(total_reward)
    
    results_lr.append({
        'param': lr,
        'rewards': cumulative_rewards,
        'mean_final': np.mean(cumulative_rewards[-10:]),
        'std_final': np.std(cumulative_rewards[-10:])
    })
    env.close()

# Visualisation Learning Rate
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for res in results_lr:
    plt.plot(res['rewards'], label=f"α={res['param']}", alpha=0.7)
plt.title('Influence du Learning Rate (α)', fontsize=14, fontweight='bold')
plt.xlabel('Épisode')
plt.ylabel('Retour cumulé')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
params = [res['param'] for res in results_lr]
means = [res['mean_final'] for res in results_lr]
stds = [res['std_final'] for res in results_lr]
plt.errorbar(params, means, yerr=stds, marker='o', capsize=5, linewidth=2)
plt.title('Performance finale vs Learning Rate', fontsize=14, fontweight='bold')
plt.xlabel('Learning Rate (α)')
plt.ylabel('Retour moyen (10 derniers épisodes)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_qlearning/learning_rate_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# TEST 2: Influence du Discount Factor (γ)
# =============================================================================
print("\n" + "="*60)
print("TEST 2: Influence du Discount Factor (γ)")
print("="*60)

results_gamma = []
for gamma in discount_factors:
    print(f"\n→ Test avec γ = {gamma}")
    env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)
    agent = QLearningAgent(env, discount_factor=gamma)
    
    cumulative_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(tuple(state), action, reward, tuple(next_state))
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        cumulative_rewards.append(total_reward)
    
    results_gamma.append({
        'param': gamma,
        'rewards': cumulative_rewards,
        'mean_final': np.mean(cumulative_rewards[-10:]),
        'std_final': np.std(cumulative_rewards[-10:])
    })
    env.close()

# Visualisation Discount Factor
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for res in results_gamma:
    plt.plot(res['rewards'], label=f"γ={res['param']}", alpha=0.7)
plt.title('Influence du Discount Factor (γ)', fontsize=14, fontweight='bold')
plt.xlabel('Épisode')
plt.ylabel('Retour cumulé')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
params = [res['param'] for res in results_gamma]
means = [res['mean_final'] for res in results_gamma]
stds = [res['std_final'] for res in results_gamma]
plt.errorbar(params, means, yerr=stds, marker='o', capsize=5, linewidth=2, color='green')
plt.title('Performance finale vs Discount Factor', fontsize=14, fontweight='bold')
plt.xlabel('Discount Factor (γ)')
plt.ylabel('Retour moyen (10 derniers épisodes)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_qlearning/discount_factor_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# TEST 3: Influence du Exploration Rate (ε)
# =============================================================================
print("\n" + "="*60)
print("TEST 3: Influence du Exploration Rate (ε)")
print("="*60)

results_epsilon = []
for eps in exploration_rates:
    print(f"\n→ Test avec ε = {eps}")
    env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)
    agent = QLearningAgent(env, exploration_rate=eps, exploration_decay=1.0)  # Pas de décroissance
    
    cumulative_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(tuple(state), action, reward, tuple(next_state))
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        cumulative_rewards.append(total_reward)
    
    results_epsilon.append({
        'param': eps,
        'rewards': cumulative_rewards,
        'mean_final': np.mean(cumulative_rewards[-10:]),
        'std_final': np.std(cumulative_rewards[-10:])
    })
    env.close()

# Visualisation Exploration Rate
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for res in results_epsilon:
    plt.plot(res['rewards'], label=f"ε={res['param']}", alpha=0.7)
plt.title('Influence du Exploration Rate (ε)', fontsize=14, fontweight='bold')
plt.xlabel('Épisode')
plt.ylabel('Retour cumulé')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
params = [res['param'] for res in results_epsilon]
means = [res['mean_final'] for res in results_epsilon]
stds = [res['std_final'] for res in results_epsilon]
plt.errorbar(params, means, yerr=stds, marker='o', capsize=5, linewidth=2, color='orange')
plt.title('Performance finale vs Exploration Rate', fontsize=14, fontweight='bold')
plt.xlabel('Exploration Rate (ε)')
plt.ylabel('Retour moyen (10 derniers épisodes)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_qlearning/exploration_rate_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# TEST 4: Influence du Exploration Decay
# =============================================================================
print("\n" + "="*60)
print("TEST 4: Influence du Exploration Decay")
print("="*60)

results_decay = []
for decay in exploration_decays:
    print(f"\n→ Test avec decay = {decay}")
    env = MyGridWorld(size=size, start=start, goals=goals, obstacles=obstacles)
    agent = QLearningAgent(env, exploration_rate=0.9, exploration_decay=decay)
    
    cumulative_rewards = []
    epsilon_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(tuple(state), action, reward, tuple(next_state))
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
        cumulative_rewards.append(total_reward)
        epsilon_history.append(agent.exploration_rate)
    
    results_decay.append({
        'param': decay,
        'rewards': cumulative_rewards,
        'epsilon_history': epsilon_history,
        'mean_final': np.mean(cumulative_rewards[-10:]),
        'std_final': np.std(cumulative_rewards[-10:])
    })
    env.close()

# Visualisation Exploration Decay
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
for res in results_decay:
    plt.plot(res['rewards'], label=f"decay={res['param']}", alpha=0.7)
plt.title('Influence du Exploration Decay', fontsize=14, fontweight='bold')
plt.xlabel('Épisode')
plt.ylabel('Retour cumulé')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
for res in results_decay:
    plt.plot(res['epsilon_history'], label=f"decay={res['param']}", alpha=0.7)
plt.title('Évolution de ε au cours du temps', fontsize=14, fontweight='bold')
plt.xlabel('Épisode')
plt.ylabel('Exploration Rate (ε)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
params = [res['param'] for res in results_decay]
means = [res['mean_final'] for res in results_decay]
stds = [res['std_final'] for res in results_decay]
plt.errorbar(params, means, yerr=stds, marker='o', capsize=5, linewidth=2, color='purple')
plt.title('Performance finale vs Exploration Decay', fontsize=14, fontweight='bold')
plt.xlabel('Exploration Decay')
plt.ylabel('Retour moyen (10 derniers épisodes)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures_qlearning/exploration_decay_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# =============================================================================
# VISUALISATION GLOBALE
# =============================================================================
fig = plt.figure(figsize=(20, 12))

# Learning Rate
ax1 = plt.subplot(2, 4, 1)
for res in results_lr:
    ax1.plot(res['rewards'], label=f"α={res['param']}", alpha=0.6, linewidth=1.5)
ax1.set_title('Learning Rate (α)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Épisode')
ax1.set_ylabel('Retour cumulé')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 4, 5)
params = [res['param'] for res in results_lr]
means = [res['mean_final'] for res in results_lr]
ax2.bar(range(len(params)), means, color='steelblue', alpha=0.7)
ax2.set_xticks(range(len(params)))
ax2.set_xticklabels([f"{p}" for p in params])
ax2.set_xlabel('α')
ax2.set_ylabel('Performance finale')
ax2.grid(True, alpha=0.3, axis='y')

# Discount Factor
ax3 = plt.subplot(2, 4, 2)
for res in results_gamma:
    ax3.plot(res['rewards'], label=f"γ={res['param']}", alpha=0.6, linewidth=1.5)
ax3.set_title('Discount Factor (γ)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Épisode')
ax3.set_ylabel('Retour cumulé')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 4, 6)
params = [res['param'] for res in results_gamma]
means = [res['mean_final'] for res in results_gamma]
ax4.bar(range(len(params)), means, color='seagreen', alpha=0.7)
ax4.set_xticks(range(len(params)))
ax4.set_xticklabels([f"{p}" for p in params])
ax4.set_xlabel('γ')
ax4.set_ylabel('Performance finale')
ax4.grid(True, alpha=0.3, axis='y')

# Exploration Rate
ax5 = plt.subplot(2, 4, 3)
for res in results_epsilon:
    ax5.plot(res['rewards'], label=f"ε={res['param']}", alpha=0.6, linewidth=1.5)
ax5.set_title('Exploration Rate (ε)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Épisode')
ax5.set_ylabel('Retour cumulé')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 4, 7)
params = [res['param'] for res in results_epsilon]
means = [res['mean_final'] for res in results_epsilon]
ax6.bar(range(len(params)), means, color='darkorange', alpha=0.7)
ax6.set_xticks(range(len(params)))
ax6.set_xticklabels([f"{p}" for p in params])
ax6.set_xlabel('ε')
ax6.set_ylabel('Performance finale')
ax6.grid(True, alpha=0.3, axis='y')

# Exploration Decay
ax7 = plt.subplot(2, 4, 4)
for res in results_decay:
    ax7.plot(res['rewards'], label=f"decay={res['param']}", alpha=0.6, linewidth=1.5)
ax7.set_title('Exploration Decay', fontsize=12, fontweight='bold')
ax7.set_xlabel('Épisode')
ax7.set_ylabel('Retour cumulé')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(2, 4, 8)
params = [res['param'] for res in results_decay]
means = [res['mean_final'] for res in results_decay]
ax8.bar(range(len(params)), means, color='rebeccapurple', alpha=0.7)
ax8.set_xticks(range(len(params)))
ax8.set_xticklabels([f"{p}" for p in params])
ax8.set_xlabel('decay')
ax8.set_ylabel('Performance finale')
ax8.grid(True, alpha=0.3, axis='y')

plt.suptitle('Analyse complète des hyperparamètres de Q-Learning', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures_qlearning/global_analysis.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

# =============================================================================
# SAUVEGARDE DES RÉSULTATS EN CSV
# =============================================================================
with open('results/qlearning_params_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Paramètre', 'Valeur', 'Retour Moyen', 'Écart-type'])
    
    for res in results_lr:
        writer.writerow(['Learning Rate', res['param'], f"{res['mean_final']:.2f}", f"{res['std_final']:.2f}"])
    
    for res in results_gamma:
        writer.writerow(['Discount Factor', res['param'], f"{res['mean_final']:.2f}", f"{res['std_final']:.2f}"])
    
    for res in results_epsilon:
        writer.writerow(['Exploration Rate', res['param'], f"{res['mean_final']:.2f}", f"{res['std_final']:.2f}"])
    
    for res in results_decay:
        writer.writerow(['Exploration Decay', res['param'], f"{res['mean_final']:.2f}", f"{res['std_final']:.2f}"])

print("\n" + "="*60)
print("✅ RÉSULTATS FINAUX")
print("="*60)

print("\n📊 Learning Rate (α):")
for res in results_lr:
    print(f"  α={res['param']:.1f} → Retour moyen: {res['mean_final']:.2f} ± {res['std_final']:.2f}")

print("\n📊 Discount Factor (γ):")
for res in results_gamma:
    print(f"  γ={res['param']:.2f} → Retour moyen: {res['mean_final']:.2f} ± {res['std_final']:.2f}")

print("\n📊 Exploration Rate (ε):")
for res in results_epsilon:
    print(f"  ε={res['param']:.1f} → Retour moyen: {res['mean_final']:.2f} ± {res['std_final']:.2f}")

print("\n📊 Exploration Decay:")
for res in results_decay:
    print(f"  decay={res['param']:.3f} → Retour moyen: {res['mean_final']:.2f} ± {res['std_final']:.2f}")

print("\n✅ Toutes les figures sont sauvegardées dans 'figures_qlearning/'")
print("✅ Le résumé CSV est sauvegardé dans 'results/qlearning_params_summary.csv'")