import unittest
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from my_sim import create_env

class TestStableBaselines(unittest.TestCase):
    def test_ppo_with_easy_env(self):
        # Create the environment
        env = create_env()

        # Wrap the environment with DummyVecEnv
        env = DummyVecEnv([lambda: env])

        # Define the DRL model
        model = PPO("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=100000)

        # Evaluate the trained model
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

        # Assert that the mean reward is not None
        self.assertIsNotNone(mean_reward)

if __name__ == '__main__':
    unittest.main()