import pickle
import numpy as np
import matplotlib.pyplot as plt
from ..util.data import Data
from ..util.math import MathOfML
from ..util.optimizer import Optimizer
from ..util.metric import Metric
from ..util.loss import Loss


class NeuralNetwork:
    def __init__(self, activation, lamb, output_size, load_path=None):
        self.activation = activation
        self.lamb = lamb
        self.dims_of_layers = [None, output_size]
        self.params = None
        self.grads = None
        if load_path:
            self.load_params(load_path)

    def init_params(self, dims_of_layers):
        self.params = {"w": np.random.randn(dims_of_layers[0], dims_of_layers[1]),
                       "b": np.zeros((1, dims_of_layers[1]))}

    def forward(self, x, w, b):
        z, linear_cache = MathOfML.linear(x, w, b)
        if self.activation == "relu":
            a, activation_cache = MathOfML.relu(z)
        elif self.activation == "leaky_relu":
            a, activation_cache = MathOfML.leaky_relu(z)
        elif self.activation == "sigmoid":
            a, activation_cache = MathOfML.sigmoid(z)
        elif self.activation == "softmax":
            a, activation_cache = MathOfML.softmax(z)
        elif self.activation == "step_function":
            a, activation_cache = MathOfML.step_function(z)
        return a, linear_cache, activation_cache

    def backward(self, da, linear_cache, activation_cache):
        if self.activation == "sigmoid":
            dz = MathOfML.sigmoid_grad(da, activation_cache)
        elif self.activation == "relu":
            dz = MathOfML.relu_gradient(da, activation_cache)
        elif self.activation == "leaky_relu":
            dz = MathOfML.leaky_relu_gradient(da, activation_cache)
        elif self.activation == "softmax":
            dz = MathOfML.softmax_gradient(da, activation_cache)
        elif self.activation == "step_function":
            dz = MathOfML.step_function_grad(da, activation_cache)
        self.grads = MathOfML.linear_grad(dz, linear_cache)

    def forward_and_backward(self, x, y, m):
        a, linear_cache, activation_cache = self.forward(x, self.params["w"], self.params["b"])
        loss = self.loss.calc_loss(y, a, m)
        dloss = self.loss.loss_grad(y, a, m)
        self.backward(dloss, linear_cache, activation_cache)
        return loss

    def compile(self, optimizer, loss, metric):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric

    def train(self, x_train, y_train, x_test, y_test, num_of_iterations, batch, epoch):
        self.dims_of_layers[0] = x_train.shape[1]
        if self.params is None:
            self.init_params(self.dims_of_layers)
        losses = []
        for i in range(epoch):
            print(f"{i + 1}번째 에포크")
            for j in range(num_of_iterations):
                m = x_train.shape[0]
                loss = self.forward_and_backward(x_train, y_train, m)
                self.params = self.optimizer.update_params(self.params, self.grads, self.lamb)
                if j % 1000 == 0:
                    losses.append(loss)
                    print(f"epoch: {epoch + 1}, loss after iteration {j}:  {loss}")
                    train_accuracy = self.metric.predict(self.params, x_train, y_train, self.forward)
                    test_accuracy = self.metric.predict(self.params, x_test, y_test, self.forward)
                    print(f"train accuracy: {train_accuracy}%")
                    print(f"test accuracy: {test_accuracy}%")

        train_accuracy = self.metric.predict(self.params, x_train, y_train, self.forward)
        test_accuracy = self.metric.predict(self.params, x_test, y_test, self.forward)
        print(f"final train accuracy: {train_accuracy}%")
        print(f"final test accuracy: {test_accuracy}%")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("num of iterations")
        plt.ylabel("loss")
        plt.title("deep neural network")
        plt.show()

    def save(self, save_path):
        save_file = open(f"{save_path}", mode="ab")
        pickle.dump(self.params, save_file)
        save_file.close()

    def load_params(self, load_path):
        read_file = open(f"{load_path}", "rb")
        self.params = pickle.load(read_file)


if __name__ == "__main__":
    np.random.seed(42)
    x_train, y_train = Data.make_linear_gaussian_data(1000, negative_mean=[1.0, 1.0],
                                                      negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                                      positive_mean=[20.0, 20.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]],
                                                      negative_mean2=[1.0, 20.0], positive_mean2=[20.0, 1.0])
    x_test, y_test = Data.make_linear_gaussian_data(100, negative_mean=[1.0, 1.0],
                                                    negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                                    positive_mean=[20.0, 20.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]],
                                                    negative_mean2=[1.0, 20.0], positive_mean2=[20.0, 1.0])

    # 1. perceptron
    perceptron = NeuralNetwork(activation='step_function', lamb=0.01, output_size=1)
    optimizer = Optimizer(optimizer="gradient_descent", learning_rate=0.001)
    loss = Loss(loss_function='mean_square_error')
    metric = Metric(metric="accuracy")
    perceptron.compile(optimizer=optimizer, loss=loss, metric=metric)
    perceptron.train(x_train, y_train, x_test, y_test,
                     num_of_iterations=5000, batch=100, epoch=10)
    perceptron.save("./perceptron")

    # 2. logistic regression
    logistic_regression = NeuralNetwork(activation='sigmoid', lamb=0.01, output_size=1)
    optimizer = Optimizer(optimizer="gradient_descent", learning_rate=0.001)
    loss = Loss(loss_function='binary_cross_entropy')
    metric = Metric(metric="accuracy")
    logistic_regression.compile(optimizer=optimizer, loss=loss, metric=metric)
    logistic_regression.train(x_train, y_train, x_test, y_test,
                              num_of_iterations=5000, batch=100, epoch=10)

    # 3. multi layer perceptron(멀티 스텝 펑션 필요함)
    logistic_regression = NeuralNetwork(activation='step_function', lamb=0.01, output_size=10)
    optimizer = Optimizer(optimizer="gradient_descent", learning_rate=0.001)
    loss = Loss(loss_function='mean_square_error')
    metric = Metric(metric="accuracy")
    logistic_regression.compile(optimizer=optimizer, loss=loss, metric=metric)
    logistic_regression.train(x_train, y_train, x_test, y_test,
                              num_of_iterations=5000, batch=100, epoch=10)

    # 4. shallow neural network
    logistic_regression = NeuralNetwork(activation='soft_max', lamb=0.01, output_size=10)
    optimizer = Optimizer(optimizer="gradient_descent", learning_rate=0.001)
    loss = Loss(loss_function='cross_entropy')
    metric = Metric(metric="accuracy")
    logistic_regression.compile(optimizer=optimizer, loss=loss, metric=metric)
    logistic_regression.train(x_train, y_train, x_test, y_test,
                              num_of_iterations=5000, batch=100, epoch=10)
