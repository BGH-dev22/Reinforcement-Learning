import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class MyGridWorld(gym.Env):
    def __init__(self, size=5, start=(0, 0), goals=[(4, 4)], obstacles=[]):
        super().__init__()
        self.size = size
        self.start = start
        self.goals = goals
        self.obstacles = obstacles
        self.state = start

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space: (x, y)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32)

        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots()
        plt.ion()  # mode interactif

    def reset(self, seed=None, options=None):
        self.state = self.start
        return np.array(self.state), {}

    def step(self, action):
        x, y = self.state

        # Mouvements
        if action == 0 and x > 0: x -= 1        # Up
        elif action == 1 and x < self.size - 1: x += 1  # Down
        elif action == 2 and y > 0: y -= 1      # Left
        elif action == 3 and y < self.size - 1: y += 1  # Right

        next_state = (x, y)

        # Vérifie si l’agent frappe un obstacle
        if next_state in self.obstacles:
            reward = -1.0
            terminated = True  # épisode perdu
        # Vérifie si l’agent atteint un goal
        elif next_state in self.goals:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False

        self.state = next_state
        return np.array(self.state), reward, terminated, False, {}

    def render(self, mode='human'):
        self.ax.clear()

        # Dessine la grille
        self.ax.set_xticks(np.arange(-0.5, self.size, 1))
        self.ax.set_yticks(np.arange(-0.5, self.size, 1))
        self.ax.grid(True)
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.invert_yaxis()

        # Dessine les goals
        for gx, gy in self.goals:
            self.ax.scatter(gy, gx, marker='*', s=300, c='red', label='Goal')

        # Dessine les obstacles
        for ox, oy in self.obstacles:
            self.ax.scatter(oy, ox, marker='s', s=200, c='gray', label='Obstacle')

        # Dessine l’agent
        sx, sy = self.state
        self.ax.scatter(sy, sx, marker='o', s=200, c='green', label='Agent')

        # Évite doublons dans la légende
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.draw()
        plt.pause(0.3)

    def close(self):
        plt.ioff()
        plt.close()
