import numpy as np
from typing import Tuple

class LinearTDAgent:
    """
    Agent utilisant l'approximation linéaire de la fonction action-valeur avec TD(0).
    Basé sur le pseudo-code de prédiction linéaire TD(0) on-policy.
    
    Q(s, a; w) ≈ wᵀ · φ(s, a)
    
    Mise à jour TD(0):
    δ = r + γ · Q(s', a'; w) - Q(s, a; w)
    w ← w + α · δ · φ(s, a)
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 exploration_min=0.01):
        """
        Initialise l'agent TD(0) avec approximation linéaire.
        
        Args:
            env: L'environnement GridWorld
            learning_rate: Taux d'apprentissage (α)
            discount_factor: Facteur de discount (γ)
            exploration_rate: Taux d'exploration initial (ε)
            exploration_decay: Décroissance de l'exploration
            exploration_min: Exploration minimale
        """
        self.env = env
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # Nombre de features
        self.n_features = self._compute_n_features()
        self.n_actions = env.action_space.n
        
        # Vecteur de poids w (partagé pour toutes les actions avec features dépendantes de l'action)
        # Dimension: n_features * n_actions (car φ(s,a) encode l'action)
        self.weights = np.zeros(self.n_features * self.n_actions)
        
        print(f"🤖 Agent TD(0) Linéaire initialisé:")
        print(f"   - Features de base: {self.n_features}")
        print(f"   - Actions: {self.n_actions}")
        print(f"   - Dimension de w: {len(self.weights)}")
        print(f"   - Learning rate (α): {learning_rate}")
        print(f"   - Discount (γ): {discount_factor}")
    
    def _compute_n_features(self):
        """Calcule le nombre de features de base."""
        # Features: [x, y, x², y², x*y, distance_goal, bias]
        return 7
    
    def extract_features(self, state: Tuple[int, int], action: int) -> np.ndarray:
        """
        Extrait le vecteur de features φ(s, a) pour l'approximation linéaire.
        
        Le vecteur φ(s, a) encode à la fois l'état et l'action:
        - Pour chaque action, on a un bloc de features
        - Seul le bloc correspondant à l'action est non-zéro
        
        Structure: φ(s, a) = [0...0, φ_base(s), 0...0]
                             └─action 0─┘└─action a─┘└─autres─┘
        
        Features de base φ_base(s):
        1. Position x normalisée
        2. Position y normalisée  
        3. x² (non-linéarité)
        4. y² (non-linéarité)
        5. x*y (interaction)
        6. Proximité au goal (1 - distance normalisée)
        7. Bias (toujours 1)
        
        Args:
            state: Position (x, y)
            action: Action choisie
            
        Returns:
            Vecteur de features φ(s, a) de dimension (n_features * n_actions)
        """
        x, y = state
        size = self.env.size
        
        # Normalisation
        x_norm = x / (size - 1) if size > 1 else 0
        y_norm = y / (size - 1) if size > 1 else 0
        
        # Distance au goal (normalisée)
        goal = self.env.goals[0] if self.env.goals else (size-1, size-1)
        max_dist = 2 * (size - 1)
        dist_goal = (abs(x - goal[0]) + abs(y - goal[1])) / max_dist if max_dist > 0 else 0
        
        # Features de base pour cet état
        base_features = np.array([
            x_norm,               # Position x
            y_norm,               # Position y
            x_norm ** 2,          # x²
            y_norm ** 2,          # y²
            x_norm * y_norm,      # Interaction x*y
            1 - dist_goal,        # Proximité au goal
            1.0                   # Bias
        ])
        
        # Créer le vecteur complet φ(s, a)
        # Seul le bloc correspondant à l'action est non-zéro
        phi = np.zeros(self.n_features * self.n_actions)
        start_idx = action * self.n_features
        end_idx = start_idx + self.n_features
        phi[start_idx:end_idx] = base_features
        
        return phi
    
    def get_q_value(self, state: Tuple[int, int], action: int) -> float:
        """
        Calcule Q(s, a; w) = wᵀ · φ(s, a)
        
        Args:
            state: État (x, y)
            action: Action
            
        Returns:
            Valeur Q estimée
        """
        phi = self.extract_features(state, action)
        return np.dot(self.weights, phi)
    
    def get_all_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """Calcule Q(s, a; w) pour toutes les actions."""
        return np.array([self.get_q_value(state, a) for a in range(self.n_actions)])
    
    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Politique ε-greedy pour choisir l'action.
        
        Args:
            state: État actuel
            
        Returns:
            Action choisie
        """
        if np.random.random() < self.exploration_rate:
            # Exploration
            return self.env.action_space.sample()
        else:
            # Exploitation: a = argmax_a Q(s, a; w)
            q_values = self.get_all_q_values(state)
            return np.argmax(q_values)
    
    def update(self, state: Tuple[int, int], action: int, 
               reward: float, next_state: Tuple[int, int], 
               done: bool = False):
        """
        Mise à jour TD(0) selon le pseudo-code:
        
        φ_s = φ(s, a)
        φ_s2 = φ(s', a')  où a' est l'action choisie dans s'
        delta = r + (0 if done else γ · wᵀ·φ_s2) - wᵀ·φ_s
        w ← w + α · delta · φ_s
        
        Args:
            state: État actuel s
            action: Action prise a
            reward: Récompense reçue r
            next_state: État suivant s'
            done: Episode terminé?
        """
        # Extraire φ(s, a)
        phi_s = self.extract_features(state, action)
        
        # Calculer Q(s, a; w) = wᵀ · φ(s, a)
        q_current = np.dot(self.weights, phi_s)
        
        # Calculer la cible
        if done:
            # Si terminal, pas de valeur future
            target = reward
        else:
            # Choisir a' pour s' selon la politique (on-policy)
            next_action = self.choose_action(next_state)
            phi_s2 = self.extract_features(next_state, next_action)
            q_next = np.dot(self.weights, phi_s2)
            target = reward + self.discount_factor * q_next
        
        # Calcul de l'erreur TD
        delta = target - q_current
        
        # Mise à jour des poids: w ← w + α · δ · φ(s, a)
        self.weights += self.learning_rate * delta * phi_s
        
        # Décroissance de l'exploration
        if done:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
    
    def get_policy(self, state: Tuple[int, int]) -> int:
        """Retourne l'action optimale selon la politique actuelle."""
        q_values = self.get_all_q_values(state)
        return np.argmax(q_values)
    
    def get_value(self, state: Tuple[int, int]) -> float:
        """Retourne V(s) = max_a Q(s, a; w)."""
        q_values = self.get_all_q_values(state)
        return np.max(q_values)
    
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
    
    def reset_exploration(self, exploration_rate=1.0):
        """Réinitialise le taux d'exploration."""
        self.exploration_rate = exploration_rate
    
    def save_weights(self, filepath: str):
        """Sauvegarde les poids."""
        np.save(filepath, self.weights)
        print(f"✅ Poids sauvegardés dans '{filepath}'")
    
    def load_weights(self, filepath: str):
        """Charge les poids."""
        self.weights = np.load(filepath)
        print(f"✅ Poids chargés depuis '{filepath}'")


class LinearTDAgentWithEligibility(LinearTDAgent):
    """
    Agent TD avec approximation linéaire et eligibility traces TD(λ).
    Extension du TD(0) pour utiliser les traces d'éligibilité.
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 exploration_min=0.01, lambda_trace=0.8):
        """
        Args:
            lambda_trace: Paramètre λ pour les traces d'éligibilité (0 = TD(0), 1 = Monte Carlo)
        """
        super().__init__(env, learning_rate, discount_factor, 
                        exploration_rate, exploration_decay, exploration_min)
        
        self.lambda_trace = lambda_trace
        # Traces d'éligibilité: même dimension que les poids
        self.eligibility_traces = np.zeros_like(self.weights)
        
        print(f"   - Lambda (λ): {lambda_trace}")
    
    def reset_traces(self):
        """Réinitialise les traces d'éligibilité (début d'épisode)."""
        self.eligibility_traces = np.zeros_like(self.weights)
    
    def update(self, state: Tuple[int, int], action: int, 
               reward: float, next_state: Tuple[int, int], 
               done: bool = False):
        """
        Mise à jour TD(λ) avec eligibility traces.
        
        φ_s = φ(s, a)
        δ = r + (0 if done else γ · Q(s', a'; w)) - Q(s, a; w)
        e ← γ·λ·e + φ_s  (traces accumulées)
        w ← w + α · δ · e
        
        Args:
            state: État actuel s
            action: Action prise a  
            reward: Récompense reçue r
            next_state: État suivant s'
            done: Episode terminé?
        """
        # Extraire φ(s, a)
        phi_s = self.extract_features(state, action)
        
        # Calculer Q(s, a; w)
        q_current = np.dot(self.weights, phi_s)
        
        # Calculer la cible
        if done:
            target = reward
        else:
            # Choisir a' pour s' selon la politique
            next_action = self.choose_action(next_state)
            q_next = self.get_q_value(next_state, next_action)
            target = reward + self.discount_factor * q_next
        
        # Erreur TD
        delta = target - q_current
        
        # Mise à jour des traces d'éligibilité: e ← γ·λ·e + φ(s, a)
        self.eligibility_traces = (
            self.discount_factor * self.lambda_trace * self.eligibility_traces + phi_s
        )
        
        # Mise à jour des poids avec les traces: w ← w + α · δ · e
        self.weights += self.learning_rate * delta * self.eligibility_traces
        
        # Décroissance exploration
        if done:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
            # Reset traces en fin d'épisode
            self.reset_traces()


def td0_linear_prediction(env, episodes=1000, alpha=0.1, gamma=0.99):
    """
    Fonction implémentant exactement le pseudo-code TD(0) pour la prédiction linéaire.
    Version simplifiée pour démonstration pédagogique.
    
    Args:
        env: Environnement
        episodes: Nombre d'épisodes
        alpha: Learning rate
        gamma: Discount factor
        
    Returns:
        v: Vecteur de poids final
    """
    # Déterminer la dimension d
    d = 7 * env.action_space.n  # Features par action
    
    # Initialiser v = 0
    v = np.zeros(d)
    
    print(f"📚 TD(0) Linear Prediction - Version Pseudo-code")
    print(f"   Episodes: {episodes}, α={alpha}, γ={gamma}, d={d}")
    
    for episode in range(episodes):
        # s, done = env.reset(), False
        s, _ = env.reset()
        done = False
        
        while not done:
            # Choisir action selon politique (ici ε-greedy simplifiée)
            if np.random.random() < 0.1:
                a = env.action_space.sample()
            else:
                # Calculer Q pour chaque action
                q_vals = []
                for act in range(env.action_space.n):
                    phi = extract_phi_simple(s, act, env)
                    q_vals.append(np.dot(v, phi))
                a = np.argmax(q_vals)
            
            # s2, r, done = env.step(a)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            # phi_s = phi(s, a)
            phi_s = extract_phi_simple(tuple(s), a, env)
            
            # phi_s2 = phi(s2, a') où a' selon politique
            if not done:
                q_vals_next = []
                for act in range(env.action_space.n):
                    phi_next = extract_phi_simple(tuple(s2), act, env)
                    q_vals_next.append(np.dot(v, phi_next))
                a_prime = np.argmax(q_vals_next)
                phi_s2 = extract_phi_simple(tuple(s2), a_prime, env)
            else:
                phi_s2 = np.zeros_like(phi_s)
            
            # delta = r + (0 if done else gamma * np.dot(v, phi_s2)) - np.dot(v, phi_s)
            delta = r + (0 if done else gamma * np.dot(v, phi_s2)) - np.dot(v, phi_s)
            
            # v += alpha * delta * phi_s
            v += alpha * delta * phi_s
            
            # s = s2
            s = s2
        
        if (episode + 1) % 200 == 0:
            print(f"   Episode {episode + 1}/{episodes} terminé")
    
    return v


def extract_phi_simple(state, action, env):
    """Helper pour extraire φ(s, a) dans la version pseudo-code."""
    x, y = state
    size = env.size
    n_features = 7
    n_actions = env.action_space.n
    
    x_norm = x / (size - 1) if size > 1 else 0
    y_norm = y / (size - 1) if size > 1 else 0
    
    goal = env.goals[0] if env.goals else (size-1, size-1)
    max_dist = 2 * (size - 1)
    dist_goal = (abs(x - goal[0]) + abs(y - goal[1])) / max_dist if max_dist > 0 else 0
    
    base_features = np.array([
        x_norm, y_norm, x_norm**2, y_norm**2, 
        x_norm * y_norm, 1 - dist_goal, 1.0
    ])
    
    phi = np.zeros(n_features * n_actions)
    start_idx = action * n_features
    phi[start_idx:start_idx + n_features] = base_features
    
    return phi