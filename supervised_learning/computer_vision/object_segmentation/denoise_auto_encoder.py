import os
import functools
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from plot_utils import plot_image_grid
from keras_custom_callbacks import SimpleLogCallback, TensorBoardImageGridCallback

batch_size = 32
num_epochs = 50
random_seed = 42

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
img_height, img_width = x_train.shape[1:]
img_channels = 1
input_shape = (img_height, img_width, img_channels)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Even though we will use again a basic fully-coonected network,
# we need to preserve the image format of the sample this time,
# to use the Keras image pre-processing tool to add noise.
# Therefore, the augmented images will be flattened by the network itself
# via a initial Flatten() layer.
x_train = x_train.reshape((-1, img_height, img_width, img_channels))
x_test = x_test.reshape((-1, img_height, img_width, img_channels))
print("Shape of training set: {}".format(x_train.shape))
print("Shape of testing set: {}".format(x_test.shape))

batch_size = 64
train_steps_per_epoch = len(x_train) // batch_size
val_steps_per_epoch = len(x_test) // batch_size

# 모델
code_size = 32
inputs = Input(shape=input_shape, name='input')
inputs_flat = Flatten()(inputs)

# Encoding layers:
enc_1 = Dense(128, activation='relu', name='enc_dense1')(inputs_flat)
enc_2 = Dense(64, activation='relu', name='enc_dense2')(enc_1)
code = Dense(code_size, activation='relu', name='enc_dense3')(enc_2)

# Decoding layers:
dec_1 = Dense(64, activation='relu', name='dec_dense1')(code)
dec_2 = Dense(128, activation='relu', name='dec_dense2')(dec_1)
decoded = Dense(np.prod(input_shape), activation='sigmoid', name='dec_dense3')(dec_2)
decoded_reshape = Reshape(input_shape)(decoded)

autoencoder = Model(inputs, decoded_reshape)
autoencoder.summary()


# Generator of noisy images
def add_noise(img, min_noise_factor=.3, max_noise_factor=.6):
    """
    Add some random noise to an image, from a uniform distribution.
    :param img:               Image to corrupt
    :param min_noise_factor:  Min. value for the noise random average amplitude
    :param max_noise_factor:  Max. value for the noise random average amplitude
    :return:                  Corrupted image
    """
    # Generating and applying noise to image:
    noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
    img_noisy = img + noise

    # Making sure the image value are still in the proper range:
    img_noisy = np.clip(img_noisy, 0., 1.)

    return img_noisy


# 이미지에 노이즈가 잘 추가되는지 확인해보자
num_show = 12
random_image_indices = np.random.choice(len(x_test), size=num_show)

orig_samples = x_test[random_image_indices]
noisy_samples = add_noise(orig_samples)

fig = plot_image_grid([np.squeeze(orig_samples), np.squeeze(noisy_samples)],
                      grayscale=True, transpose=True)
fig.show()

# We now have a choice.
# We could simply apply our noise function to the whole training set
# and pass it to our model for training

# this solution has one inconvenient
# each original image have only one noisy version
# There is thus a risk that it may overfit some of the data.

# Another solution would be to corrupt each batch of images at each iteration
# thus creating different corrupted versions each time
# this solution provides our network with new samples each time making it more robust

# To implement it, we will use a generator
# Keras models can be trained directly on datasets (model.fit(...))
# or on generators (model.fit_generator(...))

# Though less advanced that TensorFlow tf.data.Dataset, generators share some common advantages
# for datasets too big to be loaded at once, a generator can be used to load only the images for the next batches

# Keras offers several pre-implemented generators,
# to iterate over image folders, image arrays, etc.
# Here, we will use ImageDataGenerator, which can iterate over numpy arrays to generate batches.
# This generator can also be configured to pre-process each batch before yielding it.
# We define our Keras generator, passing our noisy function as pre-processing step:
# Then we pass our dataset to the generator and specify how the yielded batch should be
train_datagen = ImageDataGenerator(preprocessing_function=add_noise)
train_generator = train_datagen.flow(x_train, x_train, batch_size=batch_size, shuffle=True)

# need to prepare the validation data too
# To be able to consistently compare metrics from one epoch to another,
# we augment the validation images with noise only once,
# and saved the resulting images so they can be reused for each epoch:

x_test_noisy = add_noise(x_test)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
val_generator = train_datagen.flow(x_test_noisy, x_test, batch_size=batch_size, shuffle=False)

# Training and Monitoring
psnr_metric = functools.partial(tf.image.psnr, max_val=1.)
psnr_metric.__name__ = 'psnr'

model_dir = os.path.join('.', 'models', 'ae_denoising_mnist')
metrics_to_print = collections.OrderedDict([("loss", "loss"),
                                            ("v-loss", "val_loss"),
                                            ("psnr", "psnr"),
                                            ("v-psnr", "val_psnr")])
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss',
                                     restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
    # Callback to simply log metrics at the end of each epoch (saving space compared to verbose=1/2):
    SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=1),
    # Callback to log some validation results as image grids into TensorBoard:
    TensorBoardImageGridCallback(
        log_dir=model_dir, input_images=noisy_samples, target_images=orig_samples,
        tag='ae_results', figsize=(len(noisy_samples) * 3, 3 * 3),
        grayscale=True, transpose=True,
        preprocess_fn=lambda img, pred, gt: (
            # Squeezing the images from H x W x 1 to H x W, otherwise Pyplot complains:
            np.squeeze(img, -1), np.squeeze(pred, -1), np.squeeze(gt, -1)))
]

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[psnr_metric])

history = autoencoder.fit_generator(train_generator,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=num_epochs,
                                    validation_data=val_generator,
                                    validation_steps=val_steps_per_epoch,
                                    verbose=0,
                                    callbacks=callbacks)

fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex='col')
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("psnr")
ax[1, 1].set_title("val-psnr")

ax[0, 0].plot(history.history['loss'])
ax[0, 1].plot(history.history['val_loss'])
ax[1, 0].plot(history.history['psnr'])
ax[1, 1].plot(history.history['val_psnr'])

predicted_samples = autoencoder.predict_on_batch(noisy_samples)

fig = plot_image_grid([np.squeeze(noisy_samples),
                       np.squeeze(predicted_samples),
                       np.squeeze(orig_samples)],
                      titles=['image', 'predicted', 'ground-truth'],
                      grayscale=True, transpose=True)
fig.show()
