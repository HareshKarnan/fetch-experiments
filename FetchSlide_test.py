from stable_baselines import TRPO
from stable_baselines.common.vec_env import DummyVecEnv

import gym


model = TRPO.load('trpo_fetchslide-v1.pkl')

done = False
env = gym.make('FetchSlide-v1')
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
env = DummyVecEnv([lambda: env])

obs = env.reset()

while not done:
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
