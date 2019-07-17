import torch
import gym
from stable_baselines.common.vec_env import DummyVecEnv
import sac


if True:
    sac.set_device(torch.device('cpu'))

    env_fn = lambda: gym.wrappers.TimeLimit(gym.make("Pendulum-v0"), 1000)
    env = DummyVecEnv([env_fn])

    qnet = sac.VanillaNet(1, sac.FCBody(env.observation_space.shape[0]+env.action_space.shape[0],
                                [256, 256])).to(sac.DEVICE)

    vnet = sac.VanillaNet(1, sac.FCBody(env.observation_space.shape[0],
                                [256, 256])).to(sac.DEVICE)

    actornet = sac.GaussianPolicyNet(env.action_space.shape[0],
                                 sac.FCBody(env.observation_space.shape[0], [256, 256])).to(sac.DEVICE)

    # agent = SACAgent(env, qnet, vnet, actornet, start_steps=1000, log_comment="first")
    agent = sac.SACAutoTempAgent(env, qnet, actornet, start_steps=1000, log_comment="autotemp")

    agent.learn(iterations=10000)
    agent.save_model(".")
    # print(agent.eval("out.gif"))
    env.close()
