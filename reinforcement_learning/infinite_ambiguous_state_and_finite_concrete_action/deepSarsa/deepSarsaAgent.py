import copy
import numpy as np
import matplotlib.pyplot as plt
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import python_project.reinforcement_learning.models.deepSarsa.environment as environment


class DeepSarsaAgent:
    def __init__(self):
        self.state_size = 15
        self.actions = [0, 1, 2, 3, 4]
        self.action_size = len(self.actions)
        self.epsilon = 0.999
        self.epsilon_discount_factor = 0.999
        self.epsilon_min = 0.001

        self.learning_rate = 0.001
        self.q_network = self.build_q_network()
        self.discount_factor = 0.99

        self.load_q_network = True
        if self.load_q_network:
            self.q_network.load_weights('./save/q_network.h5')
            self.epsilon = 0

    def build_q_network(self):
        q_network = models.Sequential()
        q_network.add(
            layers.Dense(units=15, activation='linear', kernel_initializer='he_uniform', input_dim=self.state_size))
        q_network.add(layers.LeakyReLU())
        q_network.add(layers.Dense(units=15, activation='linear', kernel_initializer='he_uniform'))
        q_network.add(layers.LeakyReLU())
        q_network.add(layers.Dense(units=15, activation='linear', kernel_initializer='he_uniform'))
        q_network.add(layers.LeakyReLU())
        q_network.add(layers.Dense(units=self.action_size, activation='linear', kernel_initializer='he_uniform'))
        q_network.summary()
        q_network.compile(optimizer=optimizers.Adam(lr=self.learning_rate), loss='mse')

        return q_network

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
            print(action)
        else:
            every_q = self.q_network.predict(state)
            action = np.argmax(every_q, axis=1)[0]
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_discount_factor

        every_q = self.q_network.predict(state)
        every_target_q = self.q_network.predict(next_state)
        if not done:
            every_q[0][action] = reward + self.discount_factor * every_target_q[0][next_action]
        else:
            every_q[0][action] = reward
        target = every_q
        self.q_network.fit(x=state, y=target, epochs=1, verbose=0)


if __name__ == "__main__":
    env = environment.Env()
    agent = DeepSarsaAgent()
    scores = []

    for episode in range(100):
        state = env.reset()
        state = np.float32(state)
        state = np.reshape(a=state, newshape=[1, agent.state_size])
        action = agent.get_action(state)
        score = 0

        done = False
        while not done:
            env.render()
            next_state, reward, done = env.step(action)
            next_state = np.float32(next_state)
            next_state = np.reshape(a=next_state, newshape=[1, agent.state_size])
            next_action = agent.get_action(next_state)
            score += reward
            agent.learn(state, action, reward, next_state, next_action, done)
            state = copy.deepcopy(next_state)
            action = copy.deepcopy(next_action)

        scores.append(score)
        print("episode : " + str(episode + 1) + " epsilon : " + str(agent.epsilon) + " score : " + str(
            score))

    plt.plot(np.squeeze(scores))
    plt.ylabel('scores')
    plt.xlabel('episodes')
    plt.show()
    agent.q_network.save_weights('./save/q_network.h5')
