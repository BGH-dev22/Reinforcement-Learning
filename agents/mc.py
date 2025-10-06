import numpy as np

class MonteCarloAgent:
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = np.zeros((env.size, env.size))
        self.returns = [[[] for _ in range(env.size)] for _ in range(env.size)]
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def generate_episode(self):
        episode = []
        state, _ = self.env.reset()
        done = False
        while not done:
            x, y = state
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.policy[x, y]
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated
        return episode

    def train(self, episodes=500):
        for _ in range(episodes):
            episode = self.generate_episode()
            G = 0
            visited = set()
            for state, action, reward in reversed(episode):
                x, y = state
                G = self.gamma * G + reward
                if (x, y) not in visited:
                    self.returns[x][y].append(G)
                    self.V[x, y] = np.mean(self.returns[x][y])
                    visited.add((x, y))
            # mettre à jour la politique
            for x in range(self.env.size):
                for y in range(self.env.size):
                    action_values = np.zeros(self.env.action_space.n)
                    for action in range(self.env.action_space.n):
                        env_copy = self.env
                        env_copy.state = (x, y)
                        next_state, reward, terminated, _, _ = env_copy.step(action)
                        nx, ny = next_state
                        action_values[action] = reward + self.gamma * self.V[nx, ny]
                    self.policy[x, y] = np.argmax(action_values)
