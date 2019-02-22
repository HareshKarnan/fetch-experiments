import gym
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

env = gym.make('FetchSlide-v1')
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
env = DummyVecEnv([lambda: env])

model = TRPO(MlpPolicy, env, verbose=1,tensorboard_log="./trpo_fetchslidev1/")
model.learn(total_timesteps=1000000)
model.save("trpo_fetchslide-v1")
print('Done. Saved Model')

