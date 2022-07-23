from reinforcement_learning_ml_training_server.valueIteration.environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.value_table = [[0.00] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

    def value_update_by_policy(self):
        updated_value_table = self.value_table

        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            updated_value_candidate_list = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                updated_value_candidate = reward + self.discount_factor * next_value
                updated_value_candidate_list.append(updated_value_candidate)
            updated_value = max(updated_value_candidate_list)
            updated_value_table[state[0]][state[1]] = round(updated_value, 2)

        self.value_table = updated_value_table

    def get_action(self, state):
        if state == [2, 2]:
            return []
        chosen_max_value = -99999
        list_of_chosen_actions = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            temp_value = reward + self.discount_factor * next_value
            if temp_value > chosen_max_value:
                list_of_chosen_actions.clear()
                list_of_chosen_actions.append(action)
                chosen_max_value = temp_value
            elif temp_value == chosen_max_value:
                list_of_chosen_actions.append(action)

        return list_of_chosen_actions


if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
