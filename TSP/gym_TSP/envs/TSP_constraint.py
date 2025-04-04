import logging
import math
from gym_TSP.envs.TSP_env import TSPEnv

logger = logging.getLogger(__name__)

class TSPConstraintEnv(TSPEnv):
    """
    TSPConstraintEnv defines a variation of the Traveling Salesman Problem 
    with additional constraints on the path selection.
    """

    def __init__(self):
        super(TSPConstraintEnv, self).__init__()
        self.old_path_cost = 0
        self.best_path_cost = float('inf')
        self.first_step = True

    def _get_reward(self):
        """
        Reward is based on minimizing the total travel distance in the TSP.
        """
        current_state = self.env.getState()
        current_path_cost = current_state["current_path_cost"]

        # Compute improvement in path cost from the last step
        if not self.first_step:
            path_cost_delta = self.old_path_cost - current_path_cost

        self.old_path_cost = current_path_cost

        reward = 0
        if not self.first_step:
            # Reward for reducing the path cost
            reward += path_cost_delta

            # Extra reward if a new best path is found
            if current_path_cost < self.best_path_cost:
                self.best_path_cost = current_path_cost
                reward += 5.0

        self.first_step = False
        return reward

    def _reset(self):
        self.old_path_cost = 0
        self.best_path_cost = float('inf')
        self.first_step = True
        return super(TSPConstraintEnv, self)._reset()
