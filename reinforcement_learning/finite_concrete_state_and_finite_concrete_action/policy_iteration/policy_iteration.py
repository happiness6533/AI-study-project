import random
from python_project.reinforcement_learning.models.policy_iteration.environment import Env, GraphicDisplay


class PolicyIteration:
    def __init__(self, env):
        self.env = env

        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.policy_table[2][2] = []

        self.value_table = [[0.00] * env.width for _ in range(env.height)]

        self.discount_factor = 0.9

    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

    def value_update_by_policy(self):
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            updated_value = 0.00
            for action in self.env.possible_actions:
                action_prob = self.get_policy(state)[action]
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)

                updated_value += action_prob * (reward + self.discount_factor * next_value)
            self.value_table[state[0]][state[1]] = round(updated_value, 2)

    def policy_update_by_value(self):
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            chosen_max_value = -99999
            index_list_of_chosen_actions = []
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp_value = reward + self.discount_factor * next_value
                if temp_value > chosen_max_value:
                    chosen_max_value = temp_value
                    index_list_of_chosen_actions.clear()
                    index_list_of_chosen_actions.append(index)
                elif temp_value == chosen_max_value:
                    index_list_of_chosen_actions.append(index)
            prob = 1 / len(index_list_of_chosen_actions)
            result = [0.0, 0.0, 0.0, 0.0]
            for index in index_list_of_chosen_actions:
                result[index] = prob
            self.policy_table[state[0]][state[1]] = result

    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index


env = Env()
policy_iteration = PolicyIteration(env)
grid_world = GraphicDisplay(policy_iteration)
grid_world.mainloop()
