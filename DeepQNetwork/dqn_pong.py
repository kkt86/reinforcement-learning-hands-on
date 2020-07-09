# Hi ther

from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = 'PongNoFrames-v4'
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99  # decay factor
BATCH_SIZE = 32  # batch size to sample from the replay buffer
REPLAY_SIZE = 10000  # maximum capacity of the replay buffer
REPLAY_START_SIZE = 10000  # number of frames to wait, before starting to replay from buffer
LEARNING_RATE = 1e-4  # learning rate for Adam
SYNC_TARGET_FRAMES = 1000  # frequency of syncronization between training and target model

EPSILON_DECAY_LAST_FRAME = 10 * 5
EPSILON_START = 1.0
EPSILON_MIN = 0.02

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.sample(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*self.buffer[indices])
        return np.array(states), \
               np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.unit8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones)
    next_states_v = torch.tensor(next_states).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable cuda')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME,
                        help='Name of the environment, default=' + DEFAULT_ENV_NAME)
    parser.add_argument('--reward', type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    defice = torch.device('cuda' if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(defice)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(defice)
