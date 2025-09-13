import gymnasium as gym
from ray import tune
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from utils.cube_env import RubicsCube
from torch import Tensor
import os

class RLAgent():

    def __init__(self, disorder_capability, max_steps):

        if not ray.is_initialized():
            ray.init()

        # Register the custom environment

        register_env("RubicsCube-v0", lambda config: RubicsCube(
            mode="quat",
            disorder_turns=disorder_capability,
            max_steps=max_steps
            )
        )

        # Configure the DQN agent
        config = (
            DQNConfig()
            .environment("RubicsCube-v0")
            .framework("torch")
            .training(
                gamma=0.99,
                lr=1e-3,
                train_batch_size=12,
                replay_buffer_config={
                    "type": "PrioritizedEpisodeReplayBuffer",
                    "capacity": 50000,      # make sure this >> train_batch_size
                    "prioritized_replay": True,
                    "alpha": 0.5,
                    "beta": 0.5,
                }
            )
            .env_runners(num_env_runners=1)
        )

        self.algo = config.build()
        self.ckpts = []

        #print(self.algo.workers.local_worker().get_policy().model)

    def train(self, lesson, epochs):

        for epoch in range(epochs):
            result = self.algo.train()
            episode_reward_mean = result["env_runners"]["episode_return_mean"]
            #print(result["env_runners"].keys())
            print(f"Iter {epoch}: mean reward={episode_reward_mean}")
            
            parent_path = os.path.abspath('.')
            path = os.path.join(parent_path, f'data/lesson_{lesson}_checkpoint_{epoch}')
            self.save(path)

    def save(self, save_path):
        path = self.algo.save_to_path(path = save_path)
        self.ckpts.append(path)
        print(f"saved algo to {path}")
        
    def load(self, path=False):
        # Restore state
        if path:
            self.algo.restore_from_path(path)
        else:
            self.algo.restore_from_path(self.ckpts[-1])
        self.prediction_module = self.algo.get_module()
        print("Agent ready to predict.")

    def act(self, state):

        # policy = self.algo.get_policy()
        response = self.prediction_module.forward_inference({'obs': Tensor([state])})
        action = response['actions'].item()

        return action
    
        # state, _ = env.reset()
        # done = False
        # total_reward = 0

        # while not done:
        #     action = self.algo.compute_single_action(state)
        #     state, reward, done, _ = env.step(action)
        #     total_reward += reward

        # print(f"Evaluation reward: {total_reward}")