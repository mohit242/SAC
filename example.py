import torch
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from sac import SACAgent
from sac.network import *

if True:
    set_device(torch.device('cpu'))

    env_fn = lambda: gym.wrappers.TimeLimit(gym.make("Pendulum-v0"), 1000)
    env = DummyVecEnv([env_fn])

    qnet = VanillaNet(1, FCBody(env.observation_space.shape[0]+env.action_space.shape[0],
                                [256, 256])).to(DEVICE)

    vnet = VanillaNet(1, FCBody(env.observation_space.shape[0],
                                [256, 256])).to(DEVICE)

    actornet = GaussianPolicyNet(env.action_space.shape[0],
                                 FCBody(env.observation_space.shape[0], [256, 256])).to(DEVICE)

    agent = SACAgent(env, qnet, vnet, actornet, start_steps=1000, log_comment="first")

    agent.learn(iterations=10000)
    # print(agent.eval("out.gif"))
    env.close()
