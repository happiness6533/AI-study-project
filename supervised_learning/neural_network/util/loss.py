import numpy as np


class Loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def regularize(self, params, loss, lamb):
        # 코드 고쳐야 됨
        w = params["w"]
        return loss + lamb * np.sum(np.square(w)) / 2

    def calc_loss(self, y, a, m):
        if self.loss_function == "binary_cross_entropy":
            return np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))) / m
        elif self.loss_function == "mean_square_error":
            return np.sum(np.square(y - a)) / (2 * m)
        elif self.loss_function == "cross_entropy":
            return np.sum(-(y * np.log(a))) / m

    def loss_grad(self, y, a, m):
        if self.loss_function == "binary_cross_entropy":
            return -(y / a - (1 - y) / (1 - a)) / m
        elif self.loss_function == "mean_square_error":
            return (a - y) / m
        elif self.loss_function == "cross_entropy":
            return -(y / a) / m


if __name__ == "__main__":
    import numpy as np
