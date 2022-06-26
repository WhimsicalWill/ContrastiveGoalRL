import gym
import numpy as np

# OpenAI gym environments that are good for goal-conditioned RL
# fetch_{reach,push}
# fetch_{reach,push}_image

env = gym.make("FetchPush-v1")

render = True
score = 0
num_episodes = 10
for ep in range(num_episodes):
    done = False
    while not done:
        obs_, reward, done, info = env.step(env.action_space.sample())
        score += reward
        if render:
            env.render()
        
