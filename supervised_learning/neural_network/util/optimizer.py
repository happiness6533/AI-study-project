import numpy as np


class Optimizer:
    def __init__(self, optimizer, learning_rate):
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def update_params(self, params, grads, lamb):
        if self.optimizer == "gradient_descent":
            params["w"] = params["w"] * (1 - lamb * self.learning_rate) - self.learning_rate * self.gradient_clip(
                grads["dw"], 1)
            params["b"] = params["b"] - self.learning_rate * self.gradient_clip(grads["db"], 1)

        return params

    def update_params_many(self, params, grads, num_of_layers, lamb):
        if self.optimizer == "gradient_descent":
            for i in range(1, num_of_layers):
                params["w" + str(i)] = params["w" + str(i)] * (
                        1 - lamb * self.learning_rate) - self.learning_rate * self.gradient_clip(
                    grads["dw" + str(i)], 10)
                params["b" + str(i)] = params["b" + str(i)] - self.learning_rate * self.gradient_clip(
                    grads["db" + str(i)],
                    10)
            return params

    @staticmethod
    def gradient_clip(grad, limit):
        if np.linalg.norm(grad) >= limit:
            grad = limit * (grad / np.linalg.norm(grad))
        return grad


if __name__ == "__main__":
    pass
