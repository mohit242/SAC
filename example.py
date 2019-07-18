import torch
import gym
from stable_baselines.common.vec_env import DummyVecEnv
import sac
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enables wandb logging")
    parser.add_argument("--gpu", action="store_const", const="cuda:0", default="cpu", help="Sets device to gpu")
    parser.add_argument("--name", "-n", type=str, default=None, help="Experiment name for logging")
    parser.add_argument("--env", "-e", type=str, default="Pendulum-v0", help="Name of gym env")
    parser.add_argument("--steps", type=int, default=10000, help="Number of learning steps")
    args = parser.parse_args()

    sac.set_device(torch.device(args.gpu))
    sac.set_seed(0)

    if args.wandb:
        import wandb
        wandb.init(project="sac", name=args.name, sync_tensorboard=True)
    env_fn = lambda: sac.ActionSACWrapper(gym.wrappers.TimeLimit(gym.make(args.env), 1000))
    env = DummyVecEnv([env_fn])

    qnet = sac.VanillaNet(1, sac.FCBody(env.observation_space.shape[0]+env.action_space.shape[0],
                                [256, 256])).to(sac.DEVICE)

    vnet = sac.VanillaNet(1, sac.FCBody(env.observation_space.shape[0],
                                [256, 256])).to(sac.DEVICE)

    actornet = sac.GaussianPolicyNet(env.action_space.shape[0],
                                 sac.FCBody(env.observation_space.shape[0], [256, 256])).to(sac.DEVICE)

    # agent = SACAgent(env, qnet, vnet, actornet, start_steps=1000, log_comment="first")
    agent = sac.SACAutoTempAgent(env, qnet, actornet, start_steps=1000, log_comment=args.name)

    agent.learn(iterations=args.steps)
    agent.save_model(wandb.run.dir if args.wandb else ".")
    # print(agent.eval("out.gif"))
    env.close()
