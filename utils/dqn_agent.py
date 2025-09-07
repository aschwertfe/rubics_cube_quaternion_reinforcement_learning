import gymnasium as gym
from ray import tune
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from utils.cube_env import RubicsCube

class RLAgent():

    def __init__(self):

        if not ray.is_initialized():
            ray.init()

        # Register the custom environment

        register_env("RubicsCube-v0", lambda config: RubicsCube(mode="quat", disorder_turns=1))

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

        #print(self.algo.workers.local_worker().get_policy().model)

    def train(self, epochs):

        for epoch in range(epochs):
            result = self.algo.train()
            episode_reward_mean = result["env_runners"]["episode_return_mean"]
            #print(result["env_runners"].keys())
            print(f"Iter {epoch}: mean reward={episode_reward_mean}")
        

    def save(self, save_path):
        self.algo.save(save_path)
        
    def load(self,path):
        # Restore state
        self.algo.restore(path)

    def act(self, state):

        # policy = self.algo.get_policy()
        
        action = self.algo.compute_actions([state])

        return action
    
        # state, _ = env.reset()
        # done = False
        # total_reward = 0

        # while not done:
        #     action = self.algo.compute_single_action(state)
        #     state, reward, done, _ = env.step(action)
        #     total_reward += reward

        # print(f"Evaluation reward: {total_reward}")