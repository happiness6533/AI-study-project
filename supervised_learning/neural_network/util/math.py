import numpy as np


class MathOfML:
    @staticmethod
    def linear(x, w, b):
        linear_cache = [x, w, b]
        z = np.matmul(x, w) + b
        return z, linear_cache

    @staticmethod
    def linear_grad(dz, linear_cache):
        [x, w, b] = linear_cache
        grads = {"dw": np.matmul(x.T, dz),
                 "db": np.mean(dz, axis=0, keepdims=True),
                 "da": np.matmul(dz, w.T)}
        return grads

    @staticmethod
    def sigmoid(z):
        activation_cache = [z]
        a = 1 / (1 + np.exp(-z))
        return a, activation_cache

    @staticmethod
    def sigmoid_grad(dloss, activation_cache):
        [z] = activation_cache
        a = 1 / (1 + np.exp(-z))
        return dloss * a * (1 - a)

    @staticmethod
    def step_function(z):
        activation_cache = [z]
        a = np.zeros(z.shape)
        a[a > 0] = 1
        return a, activation_cache

    @staticmethod
    def step_function_grad(da, activation_cache):
        [z] = activation_cache
        dz = np.zeros(z.shape)
        return da * dz

    @staticmethod
    def relu(z):
        activation_cache = [z]
        a = np.maximum(0, z)
        return a, activation_cache

    @staticmethod
    def relu_gradient(da, activation_cache):
        [z] = activation_cache
        dz = np.ones(z.shape)
        dz[z < 0] = 0
        return da * dz

    @staticmethod
    def leaky_relu(z):
        activation_cache = [z]
        a = np.maximum(0.01 * z, z)
        return a, activation_cache

    @staticmethod
    def leaky_relu_gradient(da, activation_cache):
        [z] = activation_cache
        dz = np.ones(da.shape)
        dz[z < 0] = 0.01
        return da * dz

    @staticmethod
    def softmax(z):
        activation_cache = [z]
        z = z - np.max(z, axis=1, keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return a, activation_cache

    @staticmethod
    def softmax_gradient(dloss, activation_cache):
        [z] = activation_cache
        z = z - np.max(z, axis=1, keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        dz = np.zeros(dloss.shape)
        (m, dim) = dloss.shape
        for k in range(m):
            middle_matrix = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        middle_matrix[i, j] = a[k, i] * (1 - a[k, i])
                    else:
                        middle_matrix[i, j] = -(a[k, i] * a[k, j])
            dz[k, :] = np.matmul(dloss[k, :], middle_matrix)
        return dz

    # 여기서부터 아직 안봄
    @staticmethod
    def derivated_sigmoid(y):  # sigmoid derivative function
        return y * (1 - y)

    @staticmethod
    def loss_L2(pred, target):  # L2 loss function
        return np.sum(np.square(pred - target)) / pred.shape[0]  # opt. we divide by the batch size

    @staticmethod
    def derivated_loss_L2(pred, target):  # L2 derivative function
        return 2 * (pred - target)

    @staticmethod
    def cross_entropy(pred, target):  # cross-entropy loss function
        return -np.mean(np.multiply(np.log(pred), target) + np.multiply(np.log(1 - pred), (1 - target)))

    @staticmethod
    def derivated_cross_entropy(pred, target):  # cross-entropy derivative function
        return (pred - target) / (pred * (1 - pred))


if __name__ == "__main__":
    import numpy as np
