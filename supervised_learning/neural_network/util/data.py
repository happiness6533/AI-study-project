import numpy as np
import matplotlib.pyplot as plt


class Data:
    def make_mini_natch(self, data, batch_size):
        pass

    @staticmethod
    def make_linear_gaussian_data(num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov,
                                  negative_mean2,
                                  positive_mean2):
        negative_samples1 = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov,
                                                          size=num_of_samples)
        positive_samples1 = np.random.multivariate_normal(mean=negative_mean2, cov=positive_cov,
                                                          size=num_of_samples)

        x = np.vstack((negative_samples1, positive_samples1)).astype(np.float32)
        y = np.vstack((np.zeros((num_of_samples, 1), dtype='float32'),
                       np.ones((num_of_samples, 1), dtype='float32')))

        plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
        plt.show()

        return x, y

    @staticmethod
    def make_gaussian_data(num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov, negative_mean2,
                           positive_mean2):
        half_num_of_samples = int(num_of_samples / 2)
        negative_samples1 = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov,
                                                          size=half_num_of_samples)
        negative_samples2 = np.random.multivariate_normal(mean=positive_mean, cov=negative_cov,
                                                          size=half_num_of_samples)
        positive_samples1 = np.random.multivariate_normal(mean=negative_mean2, cov=positive_cov,
                                                          size=half_num_of_samples)
        positive_samples2 = np.random.multivariate_normal(mean=positive_mean2, cov=positive_cov,
                                                          size=half_num_of_samples)

        x = np.vstack((negative_samples1, negative_samples2, positive_samples1, positive_samples2)).astype(np.float32)
        y = np.vstack((np.zeros((num_of_samples, 1), dtype='float32'),
                       np.ones((num_of_samples, 1), dtype='float32')))

        plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
        plt.show()

        return x, y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
