"""Cross-Entropy algorithm on CartPole problem

The algorithm operates in the following way:

    1) Play N episodes using the current model and environment
    2) Calculate the total reward for each episode and decide on a reward boundary. (We will use 70th percentile)
    3) Throw away all the episodes with a reward below the specified boundary
    4) Train model on the remaining episodes, where observation is an input, and action is the output
    5) Repeat from step 1., until desired output is reached

"""
from collections import namedtuple
import gym
import numpy as np
import random
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


# define model which gets as input the states, and predicts the actions
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# define containers for each step in an episode (containing observation and action taken,
# and the episode itself (containing the total reward and the steps taken during the episode)
EpisodeStep = namedtuple("EpisodeStep", field_names=['observation', 'action'])
Episode = namedtuple("Episode", field_names=['reward', 'steps'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for episode, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(list(map(lambda step: step.observation, episode.steps)))
            train_act.extend(list(map(lambda step: step.action, episode.steps)))
            elite_batch.append(episode)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == '__main__':
    random.seed(12345)
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v0"))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="-frozenlake-tweaked")

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_score_v = net(obs_v)
        loss_v = objective(action_score_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
        iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_mean, iter_no)
        writer.add_scalar("reward_mean", reward_bound, iter_no)
        if reward_mean > 0.8:
            print("Done!")
            break
    writer.close()

    # play one episode after model training
    obs = env.reset()
    while True:
        obs_v = torch.FloatTensor(obs)
        act_v = net(obs_v)
        action = np.argmax(act_v.data.numpy())
        next_obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
        obs = next_obs
    env.close()
