import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Defining the seed for some random operations:
random_seed = 42

# Setting some variables to format the logs:
log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\033[94m', '\033[92m'
log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
log_end_format = '\033[0m'

num_classes = 10
img_rows, img_cols, img_ch = 28, 28, 1
input_shape = (img_rows, img_cols, img_ch)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# to highlight the advantages of regularization,
# we will make the recognition task harder
# by artificially reducing the number of samples available for training
x_train, y_train = x_train[:200], y_train[:200]  # ... 200 training samples instead of 60,000...
print('Training data: {}'.format(x_train.shape))
print('Testing data: {}'.format(x_test.shape))

# 목표는 Training a Model with Regularization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout,
                                     BatchNormalization)

epochs = 200
batch_size = 32


@tf.function
def conv_layer(x, kernels, bias, s):
    z = tf.nn.conv2d(x, kernels, strides=[1, s, s, 1], padding='VALID')
    # Finally, applying the bias and activation function (e.g. ReLU):
    return tf.nn.relu(z + bias)


class SimpleConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, num_kernels=32, kernel_size=(3, 3), stride=1):
        """ Initialize the layer.
        :param num_kernels: Number of kernels for the convolution
        :param kernel_size: Kernel size (H x W)
        :param stride: Vertical/horizontal stride
        """
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, input_shape):
        """ Build the layer, initializing its parameters.
        This will be internally called the 1st time the layer is used.
        :param input_shape: Input shape for the layer (e.g. BxHxWxC)
        """
        num_input_ch = input_shape[-1]  # assuming shape format BHWC
        # Now we know the shape of the kernel tensor we need:
        kernels_shape = (*self.kernel_size, num_input_ch, self.num_kernels)
        # We initialize the filter values e.g. from a Glorot distribution:
        glorot_init = tf.initializers.GlorotUniform()
        self.kernels = self.add_weight(  # method to add Variables to layer
            name='kernels', shape=kernels_shape, initializer=glorot_init,
            trainable=True)  # and we make it trainable.
        # Same for the bias variable (e.g. from a normal distribution):
        self.bias = self.add_weight(
            name='bias', shape=(self.num_kernels,),
            initializer='random_normal', trainable=True)

    def call(self, inputs):
        """ Call the layer, apply its operations to the input tensor."""
        return conv_layer(inputs, self.kernels, self.bias, self.stride)

    def get_config(self):
        """
        Helper function to define the layer and its parameters.
        :return:        Dictionary containing the layer's configuration
        """
        return {'num_kernels': self.num_kernels,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'use_bias': self.use_bias}


# We will extend 위에 있는 layer class
# to add kernel/bias regularization.
# the Layer's method .add_loss() can be used for that purpose

from functools import partial


def l2_reg(coef=1e-2):
    return lambda x: tf.reduce_sum(x ** 2) * coef


# 위의 레이어를 상속받은 후 정규화 식을 모든 계층에 추가
class ConvWithRegularizers(SimpleConvolutionLayer):
    def __init__(self, num_kernels=32, kernel_size=(3, 3), stride=1,
                 kernel_regularizer=l2_reg(), bias_regularizer=None):
        super().__init__(num_kernels, kernel_size, stride)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shape):
        super().build(input_shape)
        # 더 쉬운 방법들
        # 1. 사실 아래처럼 직접 복잡하게 안하고 단순하게 build() 내부에서
        #     self.add_weight(..., regularize-self.kernel_regularier)
        #     이렇게 해도 됨
        # 2. 케라스에 사전정의된 것도 있다
        #     l1 = tf.kreas.regularizers.l1(0.01)
        #     Conv2D(..., kernel_regularizer=l1)
        #     이렇게 하면 fit에서 케라스는 자동으로 정규화항 로스를 계산한다
        if self.kernel_regularizer is not None:
            self.add_loss(partial(self.kernel_regularizer, self.kernels))
        if self.bias_regularizer is not None:
            self.add_loss(partial(self.bias_regularizer, self.bias))


conv = ConvWithRegularizers(num_kernels=32, kernel_size=(3, 3), stride=1,
                            kernel_regularizer=l2_reg(1.), bias_regularizer=l2_reg(1.))
conv.build(input_shape=tf.TensorShape((None, 28, 28, 1)))

# 여기에 추가 손실들이 배열로 담겨있다
# (즉, 계층마다 추가했던 모오든 정규화 텀들이 여기에 다 모여있음)
reg_losses = conv.losses
print('Regularization losses over kernel and bias parameters: {}'.format([loss.numpy() for loss in reg_losses]))

# Comparing with the L2 norms of its kernel and bias tensors:
kernel_norm, bias_norm = tf.reduce_sum(conv.kernels ** 2).numpy(), tf.reduce_sum(conv.bias ** 2).numpy()
print('L2 norms of kernel and bias parameters: {}'.format([kernel_norm, bias_norm]))

model = Sequential([
    Input(shape=input_shape),
    ConvWithRegularizers(kernel_regularizer=l2_reg(1.), bias_regularizer=l2_reg(1.)),
    ConvWithRegularizers(kernel_regularizer=l2_reg(1.), bias_regularizer=l2_reg(1.)),
    ConvWithRegularizers(kernel_regularizer=l2_reg(1.), bias_regularizer=l2_reg(1.))
])
print('Losses attached to the model and its layers:\n\r{} ({} losses)'.format([loss.numpy() for loss in model.losses],
                                                                              len(model.losses)))


# 이제 정규화가 된 콘볼류션 레이어를 써서 르넷을 만들고 테스트를 해보자
class LeNet5(Model):
    def __init__(self, num_classes, kernel_regularizer=l2_reg(), bias_regularizer=l2_reg()):
        super(LeNet5, self).__init__()
        self.conv1 = ConvWithRegularizers(6, kernel_size=(5, 5),
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
        self.conv2 = ConvWithRegularizers(16, kernel_size=(5, 5),
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
        self.max_pool = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(84, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.max_pool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x)))
        return x


optimizer = tf.optimizers.SGD()
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
log_string_template = 'Epoch {0:3}/{1}: main loss = {5}{2:5.3f}{6}; ' + \
                      'reg loss = {5}{3:5.3f}{6}; val acc = {5}{4:5.3f}%{6}'


def train_classifier_on_mnist(model, log_frequency=10):
    avg_main_loss = tf.keras.metrics.Mean(name='avg_main_loss', dtype=tf.float32)
    avg_reg_loss = tf.keras.metrics.Mean(name='avg_reg_loss', dtype=tf.float32)

    print("Training: {}start{}".format(log_begin_red, log_end_format))
    for epoch in range(epochs):
        for (batch_images, batch_gts) in dataset:  # For each batch of this epoch

            with tf.GradientTape() as grad_tape:
                y = model(batch_images)
                main_loss = tf.losses.sparse_categorical_crossentropy(batch_gts, y)
                reg_loss = sum(model.losses)  # 모든 계층의 정규화 로스들 추가!
                loss = main_loss + reg_loss

            grads = grad_tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            avg_main_loss.update_state(main_loss)
            avg_reg_loss.update_state(reg_loss)

        if epoch % log_frequency == 0 or epoch == (epochs - 1):  # Log some metrics
            # Validate, computing the accuracy on test data:
            acc = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(
                tf.constant(y_test), model(x_test))).numpy() * 100

            main_loss = avg_main_loss.result()
            reg_loss = avg_reg_loss.result()

            print(log_string_template.format(
                epoch, epochs, main_loss, reg_loss, acc, log_begin_blue, log_end_format))

        avg_main_loss.reset_states()
        avg_reg_loss.reset_states()
    print("Training: {}end{}".format(log_begin_green, log_end_format))
    return model


# 정규화를 한거 안한거 비교를 해봅시다
model = LeNet5(10, kernel_regularizer=l2_reg(), bias_regularizer=l2_reg())
model = train_classifier_on_mnist(model, log_frequency=10)

model = LeNet5(10, kernel_regularizer=None, bias_regularizer=None)
model = train_classifier_on_mnist(model, log_frequency=50)


# 이제 이미 케라스에 준비되어 있는 정규화 방법을 써보자
def lenet(name='lenet', input_shape=input_shape,
          use_dropout=False, use_batchnorm=False, regularizer=None):
    layers = []

    # 1st block:
    layers += [Conv2D(6, kernel_size=(5, 5), padding='same',
                      input_shape=input_shape, kernel_regularizer=regularizer)]
    if use_batchnorm:
        layers += [BatchNormalization()]
    layers += [Activation('relu'),
               MaxPooling2D(pool_size=(2, 2))]
    if use_dropout:
        layers += [Dropout(0.25)]

    # 2nd block:
    layers += [
        Conv2D(16, kernel_size=(5, 5), kernel_regularizer=regularizer)]
    if use_batchnorm:
        layers += [BatchNormalization()]
    layers += [Activation('relu'),
               MaxPooling2D(pool_size=(2, 2))]
    if use_dropout:
        layers += [Dropout(0.25)]

    # Dense layers:
    layers += [Flatten()]

    layers += [Dense(120, kernel_regularizer=regularizer)]
    if use_batchnorm:
        layers += [BatchNormalization()]
    layers += [Activation('relu')]
    if use_dropout:
        layers += [Dropout(0.25)]

    layers += [Dense(84, kernel_regularizer=regularizer)]
    layers += [Activation('relu')]

    layers += [Dense(num_classes, activation='softmax')]

    model = Sequential(layers, name=name)
    return model


# 아래의 모든 정규화 특징을 알아두자
configurations = {
    'none': {'use_dropout': False, 'use_batchnorm': False, 'regularizer': None},
    'l1': {'use_dropout': False, 'use_batchnorm': False, 'regularizer': tf.keras.regularizers.l1(0.01)},
    'l2': {'use_dropout': False, 'use_batchnorm': False, 'regularizer': tf.keras.regularizers.l2(0.01)},
    'dropout': {'use_dropout': True, 'use_batchnorm': False, 'regularizer': None},
    'bn': {'use_dropout': False, 'use_batchnorm': True, 'regularizer': None},
    'l1_dropout': {'use_dropout': False, 'use_batchnorm': True, 'regularizer': tf.keras.regularizers.l1(0.01)},
    'l1_bn': {'use_dropout': False, 'use_batchnorm': True, 'regularizer': tf.keras.regularizers.l1(0.01)},
    'l1_dropout_bn': {'use_dropout': False, 'use_batchnorm': True, 'regularizer': tf.keras.regularizers.l1(0.01)}
    # ...
}

history_per_instance = dict()

print("Experiment: {0}start{1} (training logs = off)".format(log_begin_red, log_end_format))
for config_name in configurations:
    # Resetting the seeds (for random number generation), to reduce the impact of randomness on the comparison:
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    model = lenet("lenet_{}".format(config_name), **configurations[config_name])
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Launching the training (we set `verbose=0`, so the training won't generate any logs):
    print("\t> Training with {0}: {1}start{2}".format(config_name, log_begin_red, log_end_format))
    history = model.fit(x_train, y_train, batch_size=32, epochs=300, validation_data=(x_test, y_test), verbose=0)
    history_per_instance[config_name] = history
    print('\t> Training with {0}: {1}done{2}.'.format(config_name, log_begin_green, log_end_format))

print("Experiment: {0}done{1}".format(log_begin_green, log_end_format))

fig, ax = plt.subplots(2, 2, figsize=(10, 10),
                       sharex='col')  # add parameter `sharey='row'` for a more direct comparison
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("accuracy")
ax[1, 1].set_title("val-accuracy")

lines, labels = [], []
for config_name in history_per_instance:
    history = history_per_instance[config_name]
    ax[0, 0].plot(history.history['loss'])
    ax[0, 1].plot(history.history['val_loss'])
    ax[1, 0].plot(history.history['accuracy'])
    line = ax[1, 1].plot(history.history['val_accuracy'])
    lines.append(line[0])
    labels.append(config_name)

fig.legend(lines, labels, loc='center right', borderaxespad=0.1)
plt.subplots_adjust(right=0.84)

for config_name in history_per_instance:
    best_val_acc = max(history_per_instance[config_name].history['val_accuracy']) * 100
    print('Max val-accuracy for model "{}": {:2.2f}%'.format(config_name, best_val_acc))
