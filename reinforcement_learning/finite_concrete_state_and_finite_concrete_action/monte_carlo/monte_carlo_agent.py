import numpy as np
import random
import collections
import reinforcement_learning.environment.grid_world.grid_world as grid_world


# mc + value iteration
class MonteCarloAgent:
    def __init__(self, env):
        self.actions = [0, 1, 2, 3]
        self.epsilon = 0.999
        self.epsilon_discount_factor = 0.999
        self.epsilon_min = 0.001

        self.learning_rate = 0.01
        self.value_table = collections.defaultdict(float)
        self.discount_factor = 0.9

        self.width = env.width
        self.height = env.height
        self.episode = []

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            next_values = self.possible_next_value(state)

            max_indexes = []
            max_value = next_values[0]
            for index, value in enumerate(next_values):
                if value > max_value:
                    max_value = value
                    max_indexes.clear()
                    max_indexes.append(index)
                elif value == max_value:
                    max_indexes.append(index)
            return np.random.choice(max_indexes)

    def learn(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_discount_factor
        gt = 0
        visit_state = []
        for record in reversed(self.episode):
            state = str(record[0])
            reward = record[1]
            if state not in visit_state:
                visit_state.append(state)
                gt = self.discount_factor * (reward + gt)
                old_value = self.value_table[state]
                self.value_table[state] = old_value + self.learning_rate * (gt - old_value)

    def possible_next_value(self, state):
        col, row = state
        next_values = [0.0] * 4

        if row != 0:
            next_values[0] = self.value_table[str([col, row - 1])]
        else:
            next_values[0] = self.value_table[str(state)]

        if row != self.height - 1:
            next_values[1] = self.value_table[str([col, row + 1])]
        else:
            next_values[1] = self.value_table[str(state)]

        if col != 0:
            next_values[2] = self.value_table[str([col - 1, row])]
        else:
            next_values[2] = self.value_table[str(state)]

        if col != self.width - 1:
            next_values[3] = self.value_table[str([col + 1, row])]
        else:
            next_values[3] = self.value_table[str(state)]

        return next_values

    def save_episode(self, state, reward, done):
        self.episode.append([state, reward, done])


env = grid_world.Env()
agent = MonteCarloAgent(env)
for episode in range(1000):
    state = env.reset()
    action = agent.get_action(state)

    done = False
    while not done:
        env.render()
        next_state, reward, done = env.step(action)
        agent.save_episode(next_state, reward, done)
        action = agent.get_action(next_state)
    print("episode : ", episode)
    agent.learn()
    agent.episode.clear()
