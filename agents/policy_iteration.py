import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-5):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)  # action fixe pour chaque état

    def one_step_lookahead(self, state):
        x, y = state
        action_values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            # copie de l'état pour simuler
            env_copy = self.env
            env_copy.state = (x, y)
            next_state, reward, terminated, _, _ = env_copy.step(action)
            nx, ny = next_state
            action_values[action] = reward + self.gamma * self.V[nx, ny]
        return action_values

    def policy_evaluation(self):
        while True:
            delta = 0
            for x in range(self.env.size):
                for y in range(self.env.size):
                    v = self.V[x, y]
                    a = self.policy[x, y]
                    env_copy = self.env
                    env_copy.state = (x, y)
                    next_state, reward, terminated, _, _ = env_copy.step(a)
                    nx, ny = next_state
                    self.V[x, y] = reward + self.gamma * self.V[nx, ny]
                    delta = max(delta, abs(v - self.V[x, y]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for x in range(self.env.size):
            for y in range(self.env.size):
                old_action = self.policy[x, y]
                action_values = self.one_step_lookahead((x, y))
                self.policy[x, y] = np.argmax(action_values)
                if old_action != self.policy[x, y]:
                    policy_stable = False
        return policy_stable

    def train(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
