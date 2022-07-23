import random
import collections
import copy
import numpy as np
import matplotlib.pyplot as plt
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import gym


class DqnAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.999
        self.epsilon_discount_factor = 0.9999
        self.epsilon_min = 0.001

        self.learning_rate = 0.001
        self.q_network = self.build_q_network()
        self.target_q_network = self.build_q_network()
        self.copy_q_network()
        self.memory = collections.deque(maxlen=2000)
        self.memory_min = 1000
        self.batch_size = 64
        self.discount_factor = 0.99

        self.load_q_network = True
        if self.load_q_network:
            self.q_network.load_weights('./save/q_network.h5')
            self.epsilon = 0

    def build_q_network(self):
        q_network = models.Sequential()
        q_network.add(
            layers.Dense(units=8, activation='linear', kernel_initializer='he_uniform', input_dim=self.state_size))
        q_network.add(layers.LeakyReLU())
        q_network.add(layers.Dense(units=8, activation='linear', kernel_initializer='he_uniform'))
        q_network.add(layers.LeakyReLU())
        q_network.add(layers.Dense(units=8, activation='linear', kernel_initializer='he_uniform'))
        q_network.add(layers.LeakyReLU())
        q_network.add(layers.Dense(units=self.action_size, activation='linear', kernel_initializer='he_uniform'))
        q_network.summary()
        q_network.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')
        return q_network

    def copy_q_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            every_q = self.q_network.predict(state)
            action = np.argmax(every_q, axis=1)[0]
        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def learn(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_discount_factor

        mini_batch = random.sample(population=self.memory, k=self.batch_size)
        batch_states = np.zeros(shape=(self.batch_size, self.state_size))
        batch_next_states = np.zeros_like(batch_states)
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        for i in range(self.batch_size):
            batch_states[i] = mini_batch[i][0]
            batch_actions.append(mini_batch[i][1])
            batch_rewards.append(mini_batch[i][2])
            batch_next_states[i] = mini_batch[i][3]
            batch_dones.append(mini_batch[i][4])

        batch_every_q = self.q_network.predict(batch_states)
        batch_every_target_q = self.target_q_network.predict(batch_next_states)
        for i in range(self.batch_size):
            if not batch_dones[i]:
                every_q = batch_every_q[i]
                action = batch_actions[i]
                reward = batch_rewards[i]
                every_target_q = batch_every_target_q[i]
                every_q[action] = reward + self.discount_factor * np.amax(every_target_q)
                batch_every_q[i] = every_q
            else:
                every_q = batch_every_q[i]
                action = batch_actions[i]
                reward = batch_rewards[i]
                every_q[action] = reward
                batch_every_q[i] = every_q
        batch_targets = batch_every_q
        self.q_network.fit(x=batch_states, y=batch_targets, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DqnAgent(state_size, action_size)
    scores = []

    for episode in range(1000):
        state = env.reset()
        state = np.reshape(a=state, newshape=[1, state_size])
        score = 0

        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(a=next_state, newshape=[1, state_size])
            if score < 499 and done:
                reward = -100
            score += reward
            agent.memorize(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.memory_min:
                agent.learn()
            state = copy.deepcopy(next_state)

        agent.copy_q_network()
        if score != 500:
            score += 100
        scores.append(score)
        print("episode : " + str(episode + 1) + " epsilon : " + str(agent.epsilon) + " score : " + str(
            score) + " memory : " + str(len(agent.memory)))

        if len(scores) >= 10 and np.mean(scores[-10:]) >= 499:
            plt.plot(np.squeeze(scores))
            plt.ylabel('scores')
            plt.xlabel('episodes')
            plt.show()
            agent.q_network.save_weights('./save/q_network.h5')
            break
