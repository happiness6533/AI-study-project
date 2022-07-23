import copy
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import reinforcementProject.reinforce.environment as environment


class ReinforceAgent:
    def __init__(self):
        self.load_action_network = False
        self.actions = [0, 1, 2, 3, 4]
        self.action_size = len(self.actions)
        self.state_size = 15

        self.action_network = self.build_action_network()
        self.discount_factor = 0.99
        self.learning_rate = 0.005
        self.opt = self.optimize_action_network()
        self.sampling_states = []
        self.sampling_actions = []
        self.sampling_rewards = []

        if self.load_action_network:
            self.action_network.load_weights('./save/action_network.h5')
            self.epsilon = 0.1

    # 정책 네트워크 builder
    def build_action_network(self):
        action_network = models.Sequential()
        action_network.add(layers.Dense(units=15, activation='linear', kernel_initializer='he_uniform', input_dim=self.state_size))
        action_network.add(layers.LeakyReLU())
        action_network.add(layers.Dense(units=15, activation='linear', kernel_initializer='he_uniform'))
        action_network.add(layers.LeakyReLU())
        action_network.add(layers.Dense(units=15, activation='linear', kernel_initializer='he_uniform'))
        action_network.add(layers.LeakyReLU())
        action_network.add(layers.Dense(units=self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        return action_network

    # 정책 네트워크 optimizer
    def optimize_action_network(self):
        actions = K.placeholder(shape=[None, 1])
        sampling_gt = K.placeholder(shape=[None, 1])
        loss = -K.sum(K.log(self.action_network.output[actions]) * sampling_gt)
        opt = optimizers.Adam(lr=self.learning_rate).get_updates(loss=loss)
        opt = K.function([self.action_network.input], updates=opt)
        return opt

    # 액션 >> 샘플링 >> gt >> 훈련
    def get_action(self, state):
        policy = self.action_network.predict(state)[0]
        action = np.random.choice(policy, 1, p=policy)
        return action

    def get_sample(self, state, action, reward):
        self.sampling_states.append(state)
        self.sampling_actions.append(action)
        self.sampling_rewards.append(reward)

    def get_gt(self, rewards):
        sampling_gt = np.zeros_like(rewards)
        sampling_gt[-1] = rewards[-1]
        for t in reversed(range(0, len(rewards) - 1)):
            sampling_gt[t] = rewards[t] + self.discount_factor * sampling_gt[t + 1]
        return sampling_gt

    def train_action_network(self):
        # sampling_gt 정규화
        sampling_gt = np.float32(self.get_gt(self.sampling_rewards))
        sampling_gt -= np.mean(sampling_gt)
        sampling_gt /= np.std(sampling_gt)

        # 훈련
        self.opt([self.sampling_states, self.sampling_actions, sampling_gt])

        # 초기화
        self.sampling_states = []
        self.sampling_actions = []
        self.sampling_rewards = []


if __name__ == "__main__":
    env = environment.Env()
    agent = ReinforceAgent()

    scores = []
    for episode in range(100):
        total_score = 0
        steps = 0

        state = env.reset()
        state = np.reshape(state, [1, 15])

        done = False
        while done == False:
            steps += 1
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.get_sample(state, action, reward)
            total_score += reward

            state = copy.deepcopy(next_state)

        agent.train_action_network()
        scores.append(total_score)
        print("episode : " + str(episode + 1) + " steps : " + str(steps) + " total_score : " + str(total_score))

        plt.plot(np.squeeze(scores))
        plt.ylabel('scores')
        plt.xlabel('episodes')
        plt.show()
        agent.action_network.save_weights('./save/action_network.h5')
