import numpy as np
import tensorflow as tf
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# 읽은 다음 파이썬 2와 호환되도록 디코딩합니다.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# 텍스트의 길이는 그 안에 있는 문자의 수입니다.
print ('텍스트의 길이: {}자'.format(len(text)))
# 텍스트의 처음 250자를 살펴봅니다
print(text[:250])
class CustomLstm(tf.keras.layers.Layer):
    def __init__(self, input_dim, units):
        super(CustomLstm, self).__init__()
        self.w1 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.b1 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.w2 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.b2 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.w3 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.b3 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.w4 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.b4 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.w5 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)
        self.b5 = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1),
                              trainable=True)

    def call(self, cPrev, hPrev, xt):
        concat = tf.concat([hPrev, xt], axis=0)

        forget_gate = tf.math.sigmoid(tf.linalg.matmul(self.w1, concat) + self.b1)
        update_gate = tf.math.sigmoid(tf.linalg.matmul(self.w2, concat) + self.b2)
        candidate_gate = tf.math.tanh(tf.linalg.matmul(self.w3, concat) + self.b3)
        candidate = tf.math.sigmoid(tf.linalg.matmul(self.w4, concat) + self.b4)

        ct = forget_gate * cPrev + update_gate * candidate_gate
        ht = candidate * np.tanh(ct)
        yt = tf.math.softmax(tf.linalg.matmul(self.w5, ht) + self.b5)

        return ct, ht, yt


class CustomLstmModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel1, self).__init__()
        self.lstm = CustomLstm(input_dim=784, units=32)

    def call(self, inputs, training=None):
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wy"].shape

        cStore = np.zeros((n_a, m, T_x))
        hStore = np.zeros((n_a, m, T_x))
        yStore = np.zeros((n_y, m, T_x))

        for i in range(10):
            inputs = self.lstm(c_prev, h_prev, x[:, :, t])
            cStore[:, :, t] = ct
            hStore[:, :, t] = ht
            yStore[:, :, t] = yt

            c_prev = ct
            h_prev = ht
        return cStore, hStore, yStore
