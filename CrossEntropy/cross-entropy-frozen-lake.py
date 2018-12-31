from collections import namedtuple
import gym
import numpy as np
import random
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

env = gym.make("FrozenLake-v0")

STATE_SIZE = 16
HIDDEN_SIZE = 128
BATCH_SIZE = 16


class Net(nn.Module):
    def __init__(self, state_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax()
        )

    def forward(self, x):
        return self.net(x)


class Agent(object):
    def __init__(self, state_size, hidden_size, n_actions):

        self.state_size = state_size
        self.n_actions = n_actions
        self.net = Net(state_size, hidden_size, n_actions)
        self.encoder = LabelBinarizer()
        self.encoder.fit(range(state_size))

        self.epsilon = 1.
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.005

        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state_ = torch.FloatTensor(self.encoder.transform([state])[0, :])
            action_prob = self.net(state_)
            return np.argmax(action_prob.data.numpy())

    def train_agent(self, batch):
        average_duration = np.mean(list(map(lambda episode: episode.duration, batch)))

        train_states, train_actions = [], []
        for iter_no, episode in enumerate(batch):
            train_states.extend(list(map(lambda step: self.encoder.transform([step.state])[0, :], episode.steps)))
            train_actions.extend(list(map(lambda step: step.action, episode.steps)))

        train_states = torch.FloatTensor(train_states)
        train_actions = torch.LongTensor(train_actions)
        action_scores = self.net(train_states)

        loss = self.objective(action_scores, train_actions)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return (average_duration, loss.item())


# define function which creates batch of episodes, all finished with success
def create_batch(env, agent, batch_size):
    batch = []
    while True:
        steps = []
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            steps.append(EpisodeStep(state=state, action=action))
            if done:
                if reward > 0:
                    batch.append(Episode(steps=steps, duration=len(steps)))
                break
            state = next_state
        if len(batch) == batch_size:
            return batch


EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])
Episode = namedtuple('Episode', field_names=['steps', 'duration'])

STATE_SIZE = 16
HIDDEN_SIZE = 128
BATCH_SIZE = 16

if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")

    n_actions = env.action_space.n
    agent = Agent(STATE_SIZE, HIDDEN_SIZE, n_actions)
    writer = SummaryWriter(comment="-frozen-lake")

    for epoch in range(50):
        batch = create_batch(env, agent, BATCH_SIZE)
        average_duration, loss = agent.train_agent(batch)

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('average-duration', average_duration, epoch)
        writer.add_scalar('epsilon', agent.epsilon, epoch)

        print("Epoch: %d, loss: %.3f, average duration: %.3f, epsilon: %.3f" % (
            epoch, loss, average_duration, agent.epsilon))

    writer.close()

    # play one episode after model training
    reward = 0.0
    while reward < 1:
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            if done:
                break
            state = next_state
    env.close()
