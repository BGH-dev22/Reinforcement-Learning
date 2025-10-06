import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, seed=123):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        np.random.seed(seed)

        # Q-table 3D : (x, y, action)
        self.q_table = np.zeros((env.size, env.size, env.action_space.n))

    def choose_action(self, state):
        x, y = state
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        best_next = np.max(self.q_table[nx, ny])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.alpha * td_error

    def train(self, num_episodes=500):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.update(state, action, reward, next_state)
                state = next_state
