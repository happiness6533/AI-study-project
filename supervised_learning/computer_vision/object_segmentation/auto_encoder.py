import math
import functools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.manifold import TSNE

batch_size = 128
num_epochs = 60
random_seed = 42
mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()
mnist_info = mnist_builder.info
print(mnist_info)


def _prepare_data_fn(features, target='label', flatten=True,
                     return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimensions, and opt. apply some random transformations.
    :param features:              Data
    :param target                 Target/ground-truth data to be returned along the images
                                  ('label' for categorical labels, 'image' for images, or None)
    :param flatten:               Flag to flatten the images, from (28, 28, 1) to (784,)
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """

    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.

    image = features['image']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)

    if flatten:
        is_batched = len(image.shape) > 3
        flattened_shape = (-1, 784) if is_batched else (784,)
        image = tf.reshape(image, flattened_shape)

    if target is None:
        return image if return_batch_as_tuple else {'image': image}
    else:
        features['image'] = image
        return (image, features[target]) if return_batch_as_tuple else features


def get_mnist_dataset(phase='train', target='label', batch_size=32, num_epochs=None,
                      shuffle=True, flatten=True, return_batch_as_tuple=True, seed=None):
    """
    Instantiate a CIFAR-100 dataset.
    :param phase:                 Phase ('train' or 'val')
    :param target                 Target/ground-truth data to be returned along the images
                                  ('label' for categorical labels, 'image' for images, or None)
    :param batch_size:            Batch size
    :param num_epochs:            Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle:               Flag to shuffle the dataset (if True)
    :param flatten:               Flag to flatten the images, from (28, 28, 1) to (784,)
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Iterable Dataset
    """

    assert (phase == 'train' or phase == 'test')

    prepare_data_fn = functools.partial(_prepare_data_fn, return_batch_as_tuple=return_batch_as_tuple,
                                        target=target, flatten=flatten, seed=seed)

    mnist_dataset = mnist_builder.as_dataset(split=tfds.Split.TRAIN if phase == 'train' else tfds.Split.TEST)
    mnist_dataset = mnist_dataset.repeat(num_epochs)
    if shuffle:
        mnist_dataset = mnist_dataset.shuffle(10000, seed=seed)
    mnist_dataset = mnist_dataset.batch(batch_size)
    mnist_dataset = mnist_dataset.map(prepare_data_fn, num_parallel_calls=4)
    mnist_dataset = mnist_dataset.prefetch(1)

    return mnist_dataset


num_classes = mnist_info.features['label'].num_classes
num_train_imgs = mnist_info.splits['train'].num_examples
num_val_imgs = mnist_info.splits['test'].num_examples

train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

train_mnist_dataset = get_mnist_dataset(
    phase='train', target='image', batch_size=batch_size, num_epochs=num_epochs,
    shuffle=True, flatten=True, seed=random_seed)

val_mnist_dataset = get_mnist_dataset(
    phase='test', target='image', batch_size=batch_size, num_epochs=1,
    shuffle=False, flatten=True, seed=random_seed)

input_shape = mnist_info.features['image'].shape
flattened_input_shape = [np.prod(input_shape)]

# 오토 인코더 모델
code_size = 32
inputs = Input(shape=flattened_input_shape, name='input')

# Encoding layers
enc_1 = Dense(128, activation='relu', name='enc_dense1')(inputs)
enc_2 = Dense(64, activation='relu', name='enc_dense2')(enc_1)
code_layer_name = 'enc_dense3'
code = Dense(code_size, activation='relu', name=code_layer_name)(enc_2)

# Decoding layers
dec_1 = Dense(64, activation='relu', name='dec_dense1')(code)
dec_2 = Dense(128, activation='relu', name='dec_dense2')(dec_1)
decoded = Dense(flattened_input_shape[0], activation='sigmoid', name='dec_dense3')(dec_2)

# use a sigmoid for the last activation, as we want the output values to be between 0 and 1, like the input ones
autoencoder = Model(inputs, decoded)
autoencoder.summary()

# it is often convenient to define a separate model for each,
# wrapping up their respective layers.
# For the encoder,
# it is simply a matter of wrapping into a model the layers from input to code:
encoder = Model(inputs, code)
encoder.summary()

# For the decoder,
# which would take for inputs codes provided by us and return the corresponding images.
# The problem is that in our autoencoder model, the decoding layers are linked to the encoding ones.
# We want them instead to be connected to a new Input, representing our user-provided codes.
# For that, we need to fetch the decoding layers and build a new graph with them, starting from this new input
input_code = Input(shape=(code_size,), name='input_code')
num_decoder_layers = 3
dec_i = input_code
for i in range(num_decoder_layers, 0, -1):
    # We get the decoder layers from the auto-encoder models, one by one:
    dec_layer = autoencoder.layers[-i]
    # Then we apply each layer to the new data, to construct a new graph
    # with the same parameters:
    dec_i = dec_layer(dec_i)

decoder = Model(input_code, dec_i)
decoder.summary()


# 모니터링
# we will pick a metric to estimate
# how well our auto-encoder is recovering the original images.
# The Peak Signal-to-Noise Ration (PSNR) is commonly used
# as it measures the quality of a corrupted or recovered signal/image
# compared to its original version.
# The higher the value, the closer to the original image

def log_n(x, n=10):
    """
    Compute log_n(x), i.e. the log base `n` value of `x`.
    :param x:   Input tensor
    :param n:   Value of the log base
    :return:    Log result
    """
    log_e = tf.math.log(x)
    div_log_n = tf.math.log(tf.constant(n, dtype=log_e.dtype))
    return log_e / div_log_n


def psnr(img_a, img_b, max_img_value=255):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two images.
    :param img_a:           Image A
    :param img_b:           Image B
    :param max_img_value:   Maximum possible pixel value of the images
    :return:                PSNR value
    """
    mse = tf.reduce_mean((img_a - img_b) ** 2)
    return 20 * log_n(max_img_value, 10) - 10 * log_n(mse, 10)


# TensorFlow actually provides an implementation of this metric: tf.image.psnr(a, b, max_val)
# let us use our implementation for now. We wrap it to fit Keras interface for metrics
psnr_metrics = functools.partial(psnr, max_img_value=1.)
psnr_metrics.__name__ = 'psnr'


# Custom callback for monitoring
# This time, we will implement our own, inheriting from the abstract Callback class.
# This class defines an interface composed of several methods
# which will be called by Keras along the training
class DynamicPlotCallback(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        # This method will be called when the training start.
        # Therefore, we use it to initialize some elements for our Callback:
        self.logs = dict()
        self.fig, self.ax = None, None

    def on_epoch_end(self, epoch, logs={}):
        # This method will be called after each epoch.
        # Keras will call this function, providing the current epoch number,
        # and the values of the various losses/metrics for this epoch (`logs` dict).

        # We add the new log values to the list...
        for key, val in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(val)
        # ... then we plot everything:
        self._plot_logs()

    def on_train_end(self, logs={}):
        pass  # our callback does nothing special at the end of the training

    def on_epoch_begin(self, epoch, logs={}):
        pass  # ... nor at the beginning of a new epoch

    def on_batch_begin(self, batch, logs={}):
        pass  # ... nor at the beginning of a new batch

    def on_batch_end(self, batch, logs={}):
        pass  # ... nor after.

    def _plot_logs(self):
        # Method to clear the figures and draw them over with new values:
        if self.fig is None:  # First call - we initialize the figure:
            num_metrics = len(self.logs)
            self.fig, self.ax = plt.subplots(math.ceil(num_metrics / 2), 2, figsize=(10, 8))
            self.fig.show()
            self.fig.canvas.draw()

        # Plotting:
        i = 0
        for key, val in self.logs.items():
            id_vert, id_hori = i // 2, i % 2
            self.ax[id_vert, id_hori].clear()
            self.ax[id_vert, id_hori].set_title(key)
            self.ax[id_vert, id_hori].plot(val)
            i += 1

        # self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.75, bottom=0.25)
        self.fig.canvas.draw()


# Training
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[psnr_metrics])
history = autoencoder.fit(train_mnist_dataset,
                          epochs=num_epochs,
                          steps_per_epoch=train_steps_per_epoch,
                          validation_data=val_mnist_dataset,
                          validation_steps=val_steps_per_epoch,
                          verbose=0,
                          callbacks=[DynamicPlotCallback()])


# a method to show the input/target images and the auto-encoder's results
def show_pairs(samples_a, samples_b, plot_fn_a="imshow", plot_fn_b="imshow"):
    """
    Plot pairs of data
    :param samples_a:   List of samples A
    :param samples_b:   List of samples B
    :param plot_fn_a:   Name of Matplotlib function to plot A (default: "imshow")
    :param plot_fn_b:   Name of Matplotlib function to plot B (default: "imshow")
    :return:            /
    """
    assert (len(samples_a) == len(samples_b))
    num_images = len(samples_a)

    figure = plt.figure(figsize=(num_images, 2))
    grid_spec = gridspec.GridSpec(1, num_images)
    for i in range(num_images):
        grid_spec_i = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=grid_spec[i], hspace=0)

        # Drawing image A:
        ax_img = figure.add_subplot(grid_spec_i[0])
        getattr(ax_img, plot_fn_a)(samples_a[i])
        plt.gray()
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)

        # Drawing image B:
        ax_img = figure.add_subplot(grid_spec_i[1])
        getattr(ax_img, plot_fn_b)(samples_b[i])
        plt.gray()
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)

    # plt.tight_layout()
    plt.show()


# input and output visualization
num_show = 12
x_test_sample = next(val_mnist_dataset.__iter__())[0][:num_show].numpy()
x_decoded = autoencoder.predict_on_batch(x_test_sample).numpy()
show_pairs(x_test_sample.reshape(num_show, *input_shape[:2]),
           x_decoded.reshape(num_show, *input_shape[:2]))

# Code visualization
x_encoded = encoder.predict_on_batch(x_test_sample).numpy()

# We scale up the code, to better visualize it:
x_encoded_show = np.tile(x_encoded.reshape(num_show, 1, code_size), (1, 15, 1))
show_pairs(x_test_sample.reshape(num_show, *input_shape[:2]),
           x_encoded_show, plot_fn_b="matshow")

# dataset embedding with t-SNE
tsne = TSNE(n_components=2, verbose=1, random_state=0)
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], *flattened_input_shape)

x_encoded = encoder.predict_on_batch(x_test)
x_2d = tsne.fit_transform(x_encoded)

# t-sne visualization
font = {'family': 'normal', 'weight': 'bold', 'size': 22}
matplotlib.rc('font', **font)
figure = plt.figure(figsize=(15, 10))
plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y_test, cmap=plt.cm.get_cmap("jet", num_classes))
plt.colorbar(ticks=range(num_classes))
plt.clim(-0.5, num_classes - 0.5)
plt.show()
