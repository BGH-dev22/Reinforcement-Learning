import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-5):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self):
        while True:
            delta = 0
            for x in range(self.env.size):
                for y in range(self.env.size):
                    v = self.V[x, y]
                    action_values = np.zeros(self.env.action_space.n)
                    for action in range(self.env.action_space.n):
                        env_copy = self.env
                        env_copy.state = (x, y)
                        next_state, reward, terminated, _, _ = env_copy.step(action)
                        nx, ny = next_state
                        action_values[action] = reward + self.gamma * self.V[nx, ny]
                    self.V[x, y] = np.max(action_values)
                    delta = max(delta, abs(v - self.V[x, y]))
            if delta < self.theta:
                break
        # construire la politique finale
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
