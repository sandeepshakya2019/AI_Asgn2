import logging
import gym
from gym_TSP.envs.TSP_env import TSPEnv

logger = logging.getLogger(__name__)

class TSPNotConstraintEnv(gym.Env):
    """
    TSPWithoutConstraints initializes the agent with a randomly
    placed set of cities and tasks it with finding the shortest tour.

    Unlike a constrained TSP (e.g., time windows, vehicle capacity),
    this version does not impose any additional restrictions.

    The reward is based on the total travel distance.
    """

    def __init__(self, num_cities=20):
        super(TSPNotConstraintEnv, self).__init__(num_cities)

    def _configure_environment(self):
        """
        Configures the TSP environment. No additional constraints.
        """
        pass  # No special setup needed for unconstrained TSP
