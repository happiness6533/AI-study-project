import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.python.ops.numpy_ops.np_config as np_config

np_config.enable_numpy_behavior()

# 1. 데이터
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
train_X = train_X / 255.0
test_X = test_X / 255.0
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

plt.figure(figsize=(10, 10))
for c in range(16):
    plt.subplot(4, 4, c + 1)
    plt.imshow(train_X[c].reshape(28, 28), cmap='gray')
plt.show()
print(train_Y[:16])

image_generator = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    shear_range=0.5,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    vertical_flip=False)
augment_size = 10
randidx = np.random.randint(train_X.shape[0], size=augment_size)
x_augmented = train_X[randidx].copy()
y_augmented = train_Y[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                   batch_size=augment_size, shuffle=False).next()[0]
train_X = np.concatenate((train_X, x_augmented))
train_Y = np.concatenate((train_Y, y_augmented))


# 2. 레이어
class CustomConv(tf.keras.layers.Layer):
    def __init__(self, filter_size, prev_a_channel, num_of_filters, activation, pad_size, filter_stride):
        super(CustomConv, self).__init__()
        self.filter_w = tf.Variable(
            initial_value=tf.random.normal(shape=(filter_size, filter_size, prev_a_channel, num_of_filters),
                                           mean=0., stddev=1.),
            trainable=True)
        self.filter_b = tf.Variable(
            initial_value=tf.zeros(shape=(1, 1, 1, num_of_filters)),
            trainable=True)
        self.activation = tf.keras.activations.get(activation)
        self.pad_size = pad_size
        self.filter_stride = filter_stride

    def call(self, prev_a):
        (m, prev_a_height, prev_a_width, prev_a_channel) = prev_a.shape
        (filter_size, filter_size, prev_a_channel, num_of_filters) = self.filter_w.shape

        prev_a_pad = tf.pad(prev_a, tf.constant(
            [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]]))
        z_height = int((prev_a_height + 2 * self.pad_size - filter_size) / self.filter_stride) + 1
        z_width = int((prev_a_width + 2 * self.pad_size - filter_size) / self.filter_stride) + 1

        z = tf.Variable(np.zeros((m, z_height, z_width, num_of_filters), dtype='float32'))
        for i in range(m):
            for h in range(z_height):
                for w in range(z_width):
                    for c in range(num_of_filters):
                        height_start = h * self.filter_stride
                        height_end = height_start + filter_size
                        width_start = w * self.filter_stride
                        width_end = width_start + filter_size

                        prev_a_slice = prev_a_pad[i, height_start:height_end, width_start:width_end, :]
                        z[i, h, w, c].assign(tf.add(tf.math.reduce_sum(prev_a_slice * self.filter_w[:, :, :, c], keepdims=True), self.filter_b[:, :, :, c])[0][0][0])
        return self.activation(z)


class CustomPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, pool_stride, mode='max'):
        super(CustomPool, self).__init__()
        self.mode = mode
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def call(self, a):
        (m, a_height, a_width, a_channel) = a.shape

        pool_a_height = int((a_height - self.pool_size) / self.pool_stride) + 1
        pool_a_width = int((a_width - self.pool_size) / self.pool_stride) + 1
        pool_a_channel = a_channel

        pool_a = tf.Variable(np.zeros((m, pool_a_height, pool_a_width, pool_a_channel), dtype='float32'))
        for i in range(m):
            for h in range(pool_a_height):
                for w in range(pool_a_width):
                    for c in range(pool_a_channel):
                        height_start = h * self.pool_stride
                        height_end = height_start + self.pool_size
                        width_start = w * self.pool_stride
                        width_end = width_start + self.pool_size

                        a_slice = a[i, height_start:height_end, width_start:width_end, c]
                        if self.mode == "max":
                            pool_a[i, h, w, c].assign(tf.math.reduce_max(a_slice))
                        elif self.mode == "average":
                            pool_a[i, h, w, c].assign(tf.math.reduce_mean(a_slice))
        return pool_a


# 3. 모델 빌드
class CustomCnnModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomCnnModel, self).__init__()
        self.conv1_layer = CustomConv(filter_size=3, prev_a_channel=1, num_of_filters=8,
                                      activation='relu', pad_size=1, filter_stride=1)
        self.pool1_layer = CustomPool(pool_size=2, pool_stride=2, mode='max')

        self.conv2_layer = CustomConv(filter_size=3, prev_a_channel=8, num_of_filters=16,
                                      activation='relu', pad_size=1, filter_stride=1)
        self.pool2_layer = CustomPool(pool_size=2, pool_stride=2, mode='max')

        self.conv3_layer = CustomConv(filter_size=3, prev_a_channel=16, num_of_filters=32,
                                      activation='relu', pad_size=1, filter_stride=1)
        self.pool3_layer = CustomPool(pool_size=2, pool_stride=2, mode='max')

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.drop_out_layer = tf.keras.layers.Dropout(rate=0.2)
        self.dense_layer2 = tf.keras.layers.Dense(units=5, activation='softmax')

    def call(self, inputs, training=True):
        conv1 = self.conv1_layer(inputs, training=training)
        pool1 = self.pool1_layer(conv1, training=training)
        conv2 = self.conv2_layer(pool1, training=training)
        pool2 = self.pool2_layer(conv2, training=training)
        flatten = self.flatten(pool2)
        hidden1 = self.dense_layer1(flatten)
        hidden2 = self.drop_out_layer(hidden1)
        output = self.dense_layer2(hidden2)
        return output


model = CustomCnnModel()
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy = tf.keras.metrics.CategoricalAccuracy()


# 4. 모델 학습
# @tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss(y, logits)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy.update_state(y, logits)

    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = model(x)
    accuracy.update_state(y, logits)


# 5. 모델 평가
for epoch in range(5):
    print('%d번째 epoch' % (epoch + 1))
    loss_value = train_step(train_X, train_Y)
    print('단계 / loss_value: %f / accuracy: %f' % (float(loss_value), float(accuracy.result())))

# 6. 모델 배포
