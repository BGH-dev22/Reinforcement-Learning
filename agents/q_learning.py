import numpy as np
from typing import Tuple, Dict
from collections import defaultdict

class QLearningAgent:
    """
    Agent Q-Learning utilisant une Q-table pour l'apprentissage.
    
    Algorithme Q-Learning (off-policy TD control):
    Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') - Q(s, a)]
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, 
                 exploration_min=0.01):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            env: Environnement GridWorld
            learning_rate: Taux d'apprentissage α (0 < α ≤ 1)
            discount_factor: Facteur de discount γ (0 ≤ γ ≤ 1)
            exploration_rate: Taux d'exploration initial ε (0 ≤ ε ≤ 1)
            exploration_decay: Facteur de décroissance de ε (0 < decay ≤ 1)
            exploration_min: Valeur minimale de ε
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # Q-table: dictionnaire {state: {action: q_value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Nombre d'actions disponibles
        self.n_actions = env.action_space.n
        
        # Statistiques
        self.update_count = 0
        
        print(f"🤖 Agent Q-Learning initialisé:")
        print(f"   - Learning rate (α): {learning_rate}")
        print(f"   - Discount factor (γ): {discount_factor}")
        print(f"   - Exploration rate (ε): {exploration_rate}")
        print(f"   - Exploration decay: {exploration_decay}")
        print(f"   - Exploration min: {exploration_min}")
        print(f"   - Nombre d'actions: {self.n_actions}")
    
    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """
        Retourne la Q-value pour une paire (état, action).
        
        Args:
            state: État sous forme de tuple (x, y)
            action: Action (0=haut, 1=bas, 2=gauche, 3=droite)
            
        Returns:
            Q-value pour Q(state, action)
        """
        return self.q_table[state][action]
    
    def get_all_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Retourne toutes les Q-values pour un état donné.
        
        Args:
            state: État sous forme de tuple (x, y)
            
        Returns:
            Array de Q-values [Q(s,0), Q(s,1), Q(s,2), Q(s,3)]
        """
        return np.array([self.get_q_value(state, a) for a in range(self.n_actions)])
    
    def get_max_q_value(self, state: Tuple[int, int]) -> float:
        """
        Retourne la valeur maximale Q pour un état: max_a Q(s, a).
        
        Args:
            state: État sous forme de tuple (x, y)
            
        Returns:
            max_a Q(state, a)
        """
        q_values = self.get_all_q_values(state)
        return np.max(q_values)
    
    def get_best_action(self, state: Tuple[int, int]) -> int:
        """
        Retourne la meilleure action pour un état: argmax_a Q(s, a).
        
        En cas d'égalité, choisit aléatoirement parmi les meilleures actions.
        
        Args:
            state: État sous forme de tuple (x, y)
            
        Returns:
            Meilleure action
        """
        q_values = self.get_all_q_values(state)
        max_q = np.max(q_values)
        
        # Trouver toutes les actions qui ont la valeur maximale
        best_actions = np.where(q_values == max_q)[0]
        
        # Choisir aléatoirement parmi les meilleures en cas d'égalité
        return np.random.choice(best_actions)
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Choisit une action selon la stratégie ε-greedy.
        
        Avec probabilité ε: exploration (action aléatoire)
        Avec probabilité 1-ε: exploitation (meilleure action)
        
        Args:
            state: État actuel sous forme de tuple (x, y)
            
        Returns:
            Action choisie
        """
        if np.random.random() < self.exploration_rate:
            # Exploration: action aléatoire
            return self.env.action_space.sample()
        else:
            # Exploitation: meilleure action selon Q-table
            return self.get_best_action(state)
    
    def update(self, state: Tuple[int, int], action: int, 
               reward: float, next_state: Tuple[int, int], 
               done: bool = False):
        """
        Met à jour la Q-table avec la règle de Q-Learning.
        
        Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') - Q(s, a)]
        
        Si l'épisode est terminé (done=True):
        Q(s, a) ← Q(s, a) + α[r - Q(s, a)]
        
        Args:
            state: État actuel (x, y)
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant (x', y')
            done: True si l'épisode est terminé
        """
        # Q-value actuelle
        current_q = self.get_q_value(state, action)
        
        # Calcul de la cible (target)
        if done:
            # État terminal: pas de valeur future
            target = reward
        else:
            # Q-Learning: utilise max_a' Q(s', a') (off-policy)
            max_next_q = self.get_max_q_value(next_state)
            target = reward + self.discount_factor * max_next_q
        
        # Erreur TD (TD-error)
        td_error = target - current_q
        
        # Mise à jour de la Q-value
        new_q = current_q + self.learning_rate * td_error
        self.q_table[state][action] = new_q
        
        # Statistiques
        self.update_count += 1
        
        # Décroissance de l'exploration (uniquement en fin d'épisode)
        if done:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
    
    def get_policy(self, state: Tuple[int, int]) -> int:
        """
        Retourne l'action de la politique gloutonne (greedy).
        Équivalent à get_best_action, mais nom plus explicite.
        
        Args:
            state: État (x, y)
            
        Returns:
            Action optimale selon la politique actuelle
        """
        return self.get_best_action(state)
    
    def get_value(self, state: Tuple[int, int]) -> float:
        """
        Retourne la valeur d'état V(s) = max_a Q(s, a).
        
        Args:
            state: État (x, y)
            
        Returns:
            Valeur d'état V(s)
        """
        return self.get_max_q_value(state)
    
    def get_value_grid(self) -> np.ndarray:
        """
        Génère une grille complète des valeurs d'état.
        
        Returns:
            Matrice size x size avec V(s) pour chaque état
        """
        size = self.env.size
        value_grid = np.zeros((size, size))
        
        for x in range(size):
            for y in range(size):
                value_grid[x, y] = self.get_value((x, y))
        
        return value_grid
    
    def get_policy_grid(self) -> np.ndarray:
        """
        Génère une grille complète de la politique.
        
        Returns:
            Matrice size x size avec l'action optimale pour chaque état
        """
        size = self.env.size
        policy_grid = np.zeros((size, size), dtype=int)
        
        for x in range(size):
            for y in range(size):
                policy_grid[x, y] = self.get_policy((x, y))
        
        return policy_grid
    
    def reset_exploration(self, exploration_rate: float = 1.0):
        """
        Réinitialise le taux d'exploration.
        
        Args:
            exploration_rate: Nouvelle valeur de ε
        """
        self.exploration_rate = exploration_rate
        print(f"✅ Exploration rate réinitialisé à {exploration_rate}")
    
    def get_state_action_counts(self) -> Dict:
        """
        Retourne le nombre d'états et d'actions explorés.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        n_states = len(self.q_table)
        n_state_action_pairs = sum(len(actions) for actions in self.q_table.values())
        
        return {
            'n_states_visited': n_states,
            'n_state_action_pairs': n_state_action_pairs,
            'n_updates': self.update_count
        }
    
    def print_statistics(self):
        """Affiche les statistiques de l'agent."""
        stats = self.get_state_action_counts()
        print("\n📊 Statistiques de l'agent Q-Learning:")
        print(f"   - États visités: {stats['n_states_visited']}")
        print(f"   - Paires (état, action): {stats['n_state_action_pairs']}")
        print(f"   - Nombre de mises à jour: {stats['n_updates']}")
        print(f"   - Taux d'exploration actuel: {self.exploration_rate:.4f}")
    
    def save_q_table(self, filepath: str):
        """
        Sauvegarde la Q-table dans un fichier.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        import pickle
        
        # Convertir defaultdict en dict normal pour la sauvegarde
        q_table_dict = {state: dict(actions) for state, actions in self.q_table.items()}
        
        data = {
            'q_table': q_table_dict,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'exploration_min': self.exploration_min,
            'update_count': self.update_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Q-table sauvegardée dans '{filepath}'")
    
    def load_q_table(self, filepath: str):
        """
        Charge la Q-table depuis un fichier.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restaurer la Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in data['q_table'].items():
            for action, q_value in actions.items():
                self.q_table[state][action] = q_value
        
        # Restaurer les paramètres
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.exploration_rate = data['exploration_rate']
        self.exploration_decay = data['exploration_decay']
        self.exploration_min = data['exploration_min']
        self.update_count = data['update_count']
        
        print(f"✅ Q-table chargée depuis '{filepath}'")
        self.print_statistics()
    
    def get_q_table_as_array(self) -> np.ndarray:
        """
        Convertit la Q-table en array numpy pour visualisation.
        
        Returns:
            Array 3D de shape (size, size, n_actions)
        """
        size = self.env.size
        q_array = np.zeros((size, size, self.n_actions))
        
        for x in range(size):
            for y in range(size):
                state = (x, y)
                for a in range(self.n_actions):
                    q_array[x, y, a] = self.get_q_value(state, a)
        
        return q_array
    
    def __repr__(self) -> str:
        """Représentation textuelle de l'agent."""
        return (f"QLearningAgent(α={self.learning_rate}, γ={self.discount_factor}, "
                f"ε={self.exploration_rate:.4f}, states={len(self.q_table)})")


# Fonction utilitaire pour entraîner l'agent
def train_q_learning_agent(env, agent, episodes=100, verbose=True):
    """
    Entraîne un agent Q-Learning sur un environnement.
    
    Args:
        env: Environnement GridWorld
        agent: Agent QLearningAgent
        episodes: Nombre d'épisodes d'entraînement
        verbose: Afficher les progrès
        
    Returns:
        Tuple (rewards_history, steps_history)
    """
    rewards_history = []
    steps_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(tuple(state), action, reward, tuple(next_state), 
                        done=(terminated or truncated))
            
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        if verbose and (ep + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_steps = np.mean(steps_history[-10:])
            print(f"Episode {ep+1}/{episodes} - Reward: {avg_reward:.2f}, "
                  f"Steps: {avg_steps:.1f}, ε: {agent.exploration_rate:.4f}")
    
    return rewards_history, steps_history