import numpy as np


class Metric:
    def __init__(self, metric):
        self.metric = metric

    def predict(self, params, x, y, forward):
        if self.metric == "accuracy":
            a, _, _ = forward(x, params["w"], params["b"])

            prediction = np.zeros(a.shape)
            prediction[a > 0.5] = 1
            accuracy = (1 - np.mean(np.abs(prediction - y))) * 100
            return accuracy

    def predict_dnn(self, params, x, y, forward):
        if self.metric == "accuracy":
            a, _, _ = forward(x, params["w"], params["b"])

            if a.shape[1] >= 2:
                prediction = np.zeros(a.shape)
                prediction[a >= 0.5] = 1
                accuracy = np.mean(np.all(prediction == y, axis=1, keepdims=True)) * 100
            else:
                prediction = np.zeros(a.shape)
                prediction[a > 0.5] = 1
                accuracy = (1 - np.mean(np.abs(prediction - y))) * 100
                # if (accuracy == 75):
                #     print(a)
                #     print(prediction)
                #     print(y)
                #     print()
            return accuracy


if __name__ == "__main__":
    import numpy as np
