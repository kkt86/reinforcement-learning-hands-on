import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent(object):
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        """
        Obtain a smaple in form (state, action, reward, new_state) from environment
        :return: (state, action, reward, new_state)
        """
        state = self.state
        action = self.env.action_space.sample()
        next_state, reward, done, info = self.env.step(action)
        self.state = self.env.reset() if done else next_state
        return (state, action, reward, next_state)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]

            if best_value is None or action_value >= best_value:
                best_value = action_value
                best_action = action

        return best_value, best_action

    def value_update(self, state, action, reward, next_state):
        """
        Update the action-value function by the formula
        Q(s,a) = (1-alpha)*Q(s,a) + alpha*(reward + gamma*max_a'{Q(s', a')} )
        """
        best_value, best_action = self.best_value_and_action(next_state)
        old_value = self.values[(state, action)]
        update = reward + GAMMA*best_value
        self.values[(state, action)] = (1 - ALPHA)*old_value + ALPHA*update

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        return total_reward




if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-q-learning-FL-ALPHA-{}'.format(ALPHA))

    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        state, action, reward, next_state = agent.sample_env()
        agent.value_update(state, action, reward, next_state)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= float(TEST_EPISODES)

        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward update %.3f -> %.3f at %d" % (best_reward, reward, iter_no))
            best_reward = reward
        if best_reward > 0.8:
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()
