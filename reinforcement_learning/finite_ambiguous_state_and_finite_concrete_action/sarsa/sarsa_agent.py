import numpy as np
import copy
import collections
import reinforcement_learning.environment.grid_world.grid_world as grid_world


# td + value iteration
class SarsaAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]
        self.epsilon = 0.999
        self.epsilon_discount_factor = 0.999
        self.epsilon_min = 0.001

        self.learning_rate = 0.001
        self.q_table = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.discount_factor = 0.99

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            every_q = self.q_table[state]

            max_indexes = []
            max_q = every_q[0]
            for index, q in enumerate(every_q):
                if q > max_q:
                    max_q = q
                    max_indexes.clear()
                    max_indexes.append(index)
                elif q == max_q:
                    max_indexes.append(index)
            return np.random.choice(max_indexes)

    def learn(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_discount_factor
        if not done:
            q = self.q_table[state][action]
            target_q = self.q_table[next_state][next_action]
            q = q + self.learning_rate * (reward + self.discount_factor * target_q - q)
        else:
            q = reward
        self.q_table[state][action] = q


env = grid_world.Env()
agent = SarsaAgent()
for episode in range(100):
    state = env.reset()
    state = str(state)
    action = agent.get_action(state)

    done = False
    while not done:
        env.render()
        next_state, reward, done = env.step(action)
        next_state = str(next_state)
        next_action = agent.get_action(next_state)
        agent.learn(state, action, reward, next_state, next_action, done)
        state = copy.deepcopy(next_state)
        action = copy.deepcopy(next_action)
        env.print_value_all(agent.q_table)
    print("episode : " + str(episode + 1) + ", epsilon : " + str(agent.epsilon))
