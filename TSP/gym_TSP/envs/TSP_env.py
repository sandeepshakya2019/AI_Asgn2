import os
import subprocess
import time
import signal
import gym
from gym import spaces
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TSPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_cities=20):
        """
        Initializes the TSP Environment.
        """
        super().__init__()
        self.num_cities = num_cities

        # Generate random city coordinates (float32 for compatibility)
        self.cities = np.random.rand(self.num_cities, 2).astype(np.float32)

        # Track the agent's path
        self.visited = []
        self.total_distance = 0
        self.current_city = 0  # Start at city 0

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_cities * 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cities)  # Choose the next city to visit

    def _distance(self, city1, city2):
        """ Compute Euclidean distance between two cities. """
        return np.linalg.norm(self.cities[city1] - self.cities[city2])

    def _get_reward(self):
        """
        Reward is based on minimizing the total travel distance.
        The environment is solved when all cities are visited.
        """
        if len(self.visited) == self.num_cities:
            return -self.total_distance  # Negative distance (minimization)
        return 0

    def step(self, action):
        """
        Takes an action (choosing the next city to visit).
        """
        if action in self.visited:
            # Penalize revisiting cities
            reward = -10
            done = False
        else:
            # Compute distance traveled
            distance_traveled = self._distance(self.current_city, action)
            self.total_distance += distance_traveled
            self.visited.append(action)
            self.current_city = action

            # Compute reward
            reward = self._get_reward()
            done = len(self.visited) == self.num_cities

        # Return updated state, reward, done, truncated (False by default), and info
        return self.cities.flatten(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        """ Resets the environment for a new episode. """
        super().reset(seed=seed)  # Ensure seed is properly handled
        self.visited = [0]  # Start at city 0
        self.total_distance = 0
        self.current_city = 0
        return self.cities.flatten(), {}  # Return flattened observation and info dictionary

    def render(self, mode='human'):
        """ Visualizes the TSP solution (if needed). """
        pass  # You can implement a matplotlib-based visualization here

    def close(self):
        """ Closes the environment. """
        pass
