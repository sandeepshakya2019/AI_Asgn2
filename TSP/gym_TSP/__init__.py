import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TSP-v0',
    entry_point='gym_TSP.envs.TSP_env:TSPEnv',  # Check this matches your class name
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)
