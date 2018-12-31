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
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


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
    rewards = list(map(lambda episode: episode.reward, batch))
    rewards_bound = np.percentile(rewards, percentile)
    rewards_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for episode in batch:
        if episode.reward < rewards_bound:
            continue
        train_obs.extend(list(map(lambda step: step.observation, episode.steps)))
        train_act.extend(list(map(lambda step: step.action, episode.steps)))
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, rewards_bound, rewards_mean


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_score_v = net(obs_v)
        loss_v = objective(action_score_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_m, iter_no)
        writer.add_scalar("reward_mean", reward_b, iter_no)

        if reward_m > 199:
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
