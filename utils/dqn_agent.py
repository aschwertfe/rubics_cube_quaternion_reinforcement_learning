import gymnasium as gym
from ray import tune
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from utils.cube_env import RubicsCube

class RLAgent():

    def __init__(self):
        # Register the custom environment
        gym.register(id='CustomEnv-v0', entry_point='cube_env:RubicsCube')

        # Configure the DQN agent
        config = (
            DQNConfig()
            .environment("CustomEnv-v0")
            .training(replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            })
            .env_runners(num_env_runners=1)
        )

        self.algo = config.build()

    def train(self):

        self.algo.train()
        self.algo.stop()