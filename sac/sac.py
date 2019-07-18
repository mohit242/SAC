import torch.nn as nn
import os
import time
from copy import deepcopy
from collections import deque
from imageio import mimsave
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from .network import *
from .replay import *
from .utils import *

class BaseSAC(ABC):

    def __init__(self, env, policy, writer, start_steps=10000, train_after_steps=1, gradient_steps=1,
                 replay_buffer=Replay(10e5, 256), max_eps_len=10e4):
        self.env = env
        self.states = env.reset()
        self.policy = policy
        self.start_steps = start_steps
        self.train_after_steps = train_after_steps
        self.gradient_steps = gradient_steps
        self.replay_buffer = replay_buffer
        self.max_eps_len = max_eps_len
        self.step_counter = 0
        self.writer = writer

    def _train_step(self):
        if len(self.replay_buffer) < self.start_steps:
            for _ in range(self.start_steps):
                actions = [self.env.action_space.sample() for _ in range(self.env.num_envs)]
                next_states, rewards, dones, info = self.env.step(actions)
                self.replay_buffer.add_vec([self.states, actions, rewards, next_states, dones])
                self.states = next_states
            print(f"Replay buffer initialized with {len(self.replay_buffer)} random steps")

        # self.policy.eval()
        with torch.no_grad():
            actions, log_prob, _ = self.policy.sample(self.states)
        # self.policy.train()
        next_states, rewards, dones, info = self.env.step(actions)
        actions = actions.cpu().detach().numpy()
        self.replay_buffer.add_vec([self.states, actions, rewards, next_states, dones])
        self.states = next_states
        self.step_counter += 1
        # print("Updating model")
        if self.step_counter % self.train_after_steps == 0:
            for _ in range(self.gradient_steps):
                self._update_models()
        return rewards, dones

    @abstractmethod
    def _update_models(self):
        pass

    def learn(self, iterations=1e5):

        eps_rewards = deque(maxlen=100)
        eps_rewards.append(0)
        running_rewards = np.zeros(self.env.num_envs)
        start_time = time.time()
        for i in range(int(iterations)):
            rewards, dones = self._train_step()
            running_rewards += rewards
            for idx, done in enumerate(dones):
                if done:
                    eps_rewards.append(running_rewards[idx])
                    running_rewards[idx] = 0
                    self.writer.add_scalar("train/mean_reward", np.mean(eps_rewards), global_step=self.step_counter)
                    self.writer.add_scalar("train/reward", eps_rewards[-1], global_step=self.step_counter)

            if i % (iterations // 1000) == 0:
                fps = (iterations // 1000)/(time.time() - start_time)
                start_time = time.time()
                print("Steps: {:8d}\tFPS: {:4f}\tLastest Episode reward: {:4f}\tMean Rewards: {:4f}".format(i, fps,
                                                                                                eps_rewards[-1],
                                                                                                np.mean(eps_rewards)),
                      end='\r')
        print("\n")

    def _eval_step(self):
        self.policy.eval()
        _, _, actions = self.policy.sample(self.states)
        self.policy.train()
        next_states, rewards, dones, info = self.env.step(actions.cpu().detach().numpy())
        self.states = next_states
        return rewards, dones

    def eval(self, gif_path=None):
        total_reward = 0
        done = False
        frames = []
        self.env.reset()
        while not done:
            rewards, dones = self._eval_step()
            total_reward += rewards[0]
            done = dones[0]
            if gif_path is not None:
                frames.append(self.env.render(mode='rgb_array'))
        if gif_path is not None:
            mimsave(gif_path, frames)
        return total_reward

    def save_model(self, dirpath='.'):
        if not os.path.exists(dirpath):
            raise Exception("Path does not exist")
        print(f"Saving models in directory {dirpath}")
        torch.save(self.policy.cpu().state_dict(), os.path.join(dirpath, 'policy.pt'))

    def load_model(self, dirpath='.'):
        if not os.path.exists(dirpath):
            raise Exception("Path does not exist")
        print(f"Loading models from directory {dirpath}")
        self.policy.load_state_dict(torch.load(os.path.join(dirpath, 'policy.pt')))


class SACAgent(BaseSAC):

    def __init__(self, env, qnet, vnet, policy, log_dir=None, log_comment="", start_steps=10000, train_after_steps=1,
                 gradient_steps=1, gradient_clip=1, gamma=0.99, minibatch_size=256, buffer_size=10e5,
                 polyak=0.001, max_eps_len=10e4, temperature=0.1):

        writer = SummaryWriter(log_dir, comment=log_comment)
        self.qnet = [qnet, deepcopy(qnet)]
        self.qnet[1].apply(layer_init)
        self.vnet = vnet
        self.vnet_target = deepcopy(vnet)
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.minibatch_size = minibatch_size
        replay_buffer = Replay(buffer_size, minibatch_size)
        self.polyak = polyak
        self.temperature = temperature

        super().__init__(env, policy, writer, start_steps=start_steps, train_after_steps=train_after_steps,
                         gradient_steps=gradient_steps, replay_buffer=replay_buffer, max_eps_len=max_eps_len)
        self.qnet_opt = [torch.optim.Adam(q.parameters()) for q in self.qnet]
        self.vnet_opt = torch.optim.Adam(self.vnet.parameters())
        self.policy_opt = torch.optim.Adam(self.policy.parameters())

        self.step_counter = 0

    def _update_models(self):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            val_next_state = self.vnet_target(next_states)
            q_target = tensor(rewards).unsqueeze(-1) + self.gamma * tensor(1 - dones).unsqueeze(-1) * val_next_state
            actions_v, log_pi_v, _ = self.policy.sample(states)
            qvals_v = [qf(states, actions_v) for qf in self.qnet]
            value_target = torch.min(*qvals_v) - self.temperature * log_pi_v

        qvals = [qf(states, actions) for qf in self.qnet]
        mse = nn.MSELoss()
        qloss = [mse(qv, q_target) for qv in qvals]
        value_loss = mse(self.vnet(states), value_target)

        actions_p, log_pi_p, _ = self.policy.sample(states)
        qval_p = torch.min(*[qf(states, actions_p) for qf in self.qnet])
        policy_loss = ((self.temperature * log_pi_p) - qval_p).mean()
        losses = {"qval0_loss": qloss[0], "qval1_loss": qloss[1]}
        self.writer.add_scalars("train/qvalue_loss", losses, self.step_counter)
        self.writer.add_scalar("train/value_loss", value_loss, self.step_counter)
        self.writer.add_scalar("train/policy_loss", policy_loss, self.step_counter)
        # self.writer.add_histogram("train/qnet0_weights",
        #                           self.qnet[0].state_dict()['body.net.0.weight'].flatten(), self.step_counter)
        # self.writer.add_histogram("train/qnet1_weights",
        #                           self.qnet[1].state_dict()['body.net.0.weight'].flatten(), self.step_counter)
        for i in range(2):
            self.qnet_opt[i].zero_grad()
            qloss[i].backward()
            torch.nn.utils.clip_grad_norm_(self.qnet[i].parameters(), self.gradient_clip)
            self.qnet_opt[i].step()

        self.vnet_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vnet.parameters(), self.gradient_clip)
        self.vnet_opt.step()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.policy_opt.step()

        soft_update(self.vnet_target, self.vnet, self.polyak)


class SACAutoTempAgent(BaseSAC):

    def __init__(self, env, qnet, policy, log_dir=None, log_comment="", start_steps=10000, train_after_steps=1,
                 gradient_steps=1, gradient_clip=1, gamma=0.99, minibatch_size=256, buffer_size=10e5,
                 polyak=0.001, max_eps_len=10e4):

        self.env = env
        self.states = env.reset()
        writer = SummaryWriter(log_dir, comment=log_comment)
        self.qnet = [qnet, deepcopy(qnet)]
        self.qnet[1].apply(layer_init)
        self.qnet_target = deepcopy(self.qnet)
        self.policy = policy
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.minibatch_size = minibatch_size
        replay_buffer = Replay(buffer_size, minibatch_size)
        self.polyak = polyak
        self.temperature = tensor([0.])
        self.temperature.requires_grad = True
        self.target_entropy = -torch.prod(tensor(env.action_space.shape)).item()
        self.qnet_opt = [torch.optim.Adam(q.parameters()) for q in self.qnet]
        self.policy_opt = torch.optim.Adam(self.policy.parameters())
        self.temperature_opt = torch.optim.Adam([self.temperature])

        super().__init__(env, policy, writer, start_steps=start_steps, train_after_steps=train_after_steps,
                         gradient_steps=gradient_steps, replay_buffer=replay_buffer, max_eps_len=max_eps_len)

    def _update_models(self):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            next_actions, log_prob_next, _ = self.policy.sample(next_states)
            val_next_state = torch.min(*[qf(next_states, next_actions) for qf in self.qnet_target])
            qval_next = tensor((1 - dones)).unsqueeze(-1) * (val_next_state - self.temperature * log_prob_next)
            q_target = tensor(rewards).unsqueeze(-1) + self.gamma * qval_next

        qvals = [qf(states, actions) for qf in self.qnet]
        mse = nn.MSELoss()
        qloss = [mse(qv, q_target) for qv in qvals]

        actions_p, log_pi_p, _ = self.policy.sample(states)
        qval_p = torch.min(*[qf(states, actions_p) for qf in self.qnet])
        policy_loss = ((self.temperature * log_pi_p) - qval_p).mean()

        temperature_loss = - (self.temperature * (log_pi_p + self.target_entropy).detach()).mean()
        losses = {"qval0_loss": qloss[0], "qval1_loss": qloss[1]}
        self.writer.add_scalars("train/qvalue_loss", losses, self.step_counter)
        self.writer.add_scalar("train/policy_loss", policy_loss, self.step_counter)
        self.writer.add_scalar("train/temperature_loss", temperature_loss, self.step_counter)
        # self.writer.add_histogram("train/qnet0_weights",
        #                           self.qnet[0].state_dict()['body.net.0.weight'].flatten(), self.step_counter)
        # self.writer.add_histogram("train/qnet1_weights",
        #                           self.qnet[1].state_dict()['body.net.0.weight'].flatten(), self.step_counter)
        for i in range(2):
            self.qnet_opt[i].zero_grad()
            qloss[i].backward()
            torch.nn.utils.clip_grad_norm_(self.qnet[i].parameters(), self.gradient_clip)
            self.qnet_opt[i].step()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.policy_opt.step()

        self.temperature_opt.zero_grad()
        temperature_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.temperature, self.gradient_clip)
        self.temperature_opt.step()

        self.writer.add_scalar("train/temperature", self.temperature, self.step_counter)

        for i in range(2):
            soft_update(self.qnet_target[i], self.qnet[i], self.polyak)