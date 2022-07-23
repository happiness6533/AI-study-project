import tensorflow as tf
import os
from matplotlib import pyplot as plt
import math

# Some hyper-parameters:
batch_size = 32  # Images per batch (reduce/increase according to the machine's capability)
num_epochs = 300  # Max number of training epochs
random_seed = 42  # Seed for some random operations, for reproducibility

scale_factor = 4

# The following experiments are compute-heavy (large model and dataset). Make sure the procedures are performed on GPU(s)
# Preparing the Dataset
# tensorflow-datasets, we opted for the "Rock-Paper-Scissors" dataset
# composed of 3D-rendered images of hands playing the eponym game

import tensorflow_datasets as tfds

hands_builder = tfds.builder("rock_paper_scissors")
hands_builder.download_and_prepare()

print(hands_builder.info)

from plot_utils import plot_image_grid

num_show = 5

hands_val_dataset = hands_builder.as_dataset(split=tfds.Split.TEST).batch(num_show)
hands_val_dataset_iter = hands_val_dataset.skip(1).__iter__()
# ^ `.skip(1)` is called to skip the 1st batch, which is a bit "bland" in this dataset.
# We will use the 2nd batch for our illustrations, as it has more diverse hands...
batch = next(hands_val_dataset_iter)

fig = plot_image_grid([batch['image'].numpy()], titles=['image'], transpose=True)
fig.show()

# Input pipeline with tf.data
import functools


def _prepare_data_fn(features, scale_factor=4, augment=True,
                     return_batch_as_tuple=True, seed=None):
    """
    Resize image to expected dimensions, and opt. apply some random transformations.
    :param features:              Data
    :param scale_factor:          Scale factor for the task
    :param augment:               Flag to augment the images with random operations
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    :return:                      Processed data
    """

    # Tensorflow-Dataset returns batches as feature dictionaries, expected by Estimators.
    # To train Keras models, it is more straightforward to return the batch content as tuples.

    image = features['image']
    # Convert the images to float type, also scaling their values from [0, 255] to [0., 1.]:
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Computing the scaled-down shape:
    original_shape = tf.shape(image)
    original_size = original_shape[-3:-1]
    scaled_size = original_size // scale_factor
    # Just in case the original dimensions were not a multiple of `scale_factor`,
    # we slightly resize the original image so its dimensions now are
    # (to make the loss/metrics computations easier during training):
    original_size_mult = scaled_size * scale_factor

    # Opt. augmenting the image:
    if augment:
        original_shape_mult = (original_size_mult, [tf.shape(image)[-1]])
        if len(image.shape) > 3:  # batched data:
            original_shape_mult = ([tf.shape(image)[0]], *original_shape_mult)
        original_shape_mult = tf.concat(original_shape_mult, axis=0)

        # Randomly applied horizontal flip:
        image = tf.image.random_flip_left_right(image)

        # Random B/S changes:
        image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0)  # keeping pixel values in check

        # Random resize and random crop back to expected size:
        random_scale_factor = tf.random.uniform([1], minval=1.0, maxval=1.2,
                                                dtype=tf.float32, seed=seed)
        scaled_height = tf.cast(tf.multiply(tf.cast(original_size[0], tf.float32),
                                            random_scale_factor),
                                tf.int32)
        scaled_width = tf.cast(tf.multiply(tf.cast(original_size[1], tf.float32),
                                           random_scale_factor),
                               tf.int32)
        scaled_shape = tf.squeeze(tf.stack([scaled_height, scaled_width]))
        image = tf.image.resize(image, scaled_shape)
        image = tf.image.random_crop(image, original_shape, seed=seed)

    # Generating the data pair for super-resolution task,
    # i.e. the downscaled image + its original version
    image_downscaled = tf.image.resize(image, scaled_size)

    # Just in case the original dimensions were not a multiple of `scale_factor`,
    # we slightly resize the original image so its dimensions now are
    # (to make the loss/metrics computations easier during training):
    original_size_mult = scaled_size * scale_factor
    image = tf.image.resize(image, original_size_mult)

    features = (image_downscaled, image) if return_batch_as_tuple else {'image': image_downscaled,
                                                                        'label': image}
    return features


def get_hands_dataset_for_superres(phase='train', scale_factor=4, batch_size=32, num_epochs=None,
                                   shuffle=True, augment=False, return_batch_as_tuple=True, seed=None):
    """
    Instantiate a CIFAR-100 dataset.
    :param phase:                 Phase ('train' or 'val')
    :param scale_factor:          Scale factor for the task
    :param batch_size:            Batch size
    :param num_epochs:            Number of epochs (to repeat the iteration - infinite if None)
    :param shuffle:               Flag to shuffle the dataset (if True)
    :param augment:               Flag to augment the images with random operations
    :param return_batch_as_tuple: Flag to return the batch data as tuple instead of dict
    :param seed:                  Seed for random operations
    """

    assert (phase == 'train' or phase == 'test')

    prepare_data_fn = functools.partial(
        _prepare_data_fn, scale_factor=scale_factor, augment=augment,
        return_batch_as_tuple=return_batch_as_tuple, seed=seed)

    superres_dataset = hands_builder.as_dataset(
        split=tfds.Split.TRAIN if phase == 'train' else tfds.Split.TEST)
    superres_dataset = superres_dataset.repeat(num_epochs)
    if shuffle:
        superres_dataset = superres_dataset.shuffle(
            hands_builder.info.splits[phase].num_examples, seed=seed)
    superres_dataset = superres_dataset.batch(batch_size)
    superres_dataset = superres_dataset.map(prepare_data_fn, num_parallel_calls=4)
    superres_dataset = superres_dataset.prefetch(1)

    return superres_dataset


# We now initialize our training and validation input pipelines accordingly:
# Number of images:
num_train_imgs = hands_builder.info.splits['train'].num_examples
num_val_imgs = hands_builder.info.splits['test'].num_examples

train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

# Input shape:
input_shape = hands_builder.info.features['image'].shape

# Datasets:
train_hands_dataset = get_hands_dataset_for_superres(
    phase='train', scale_factor=scale_factor, batch_size=batch_size, num_epochs=num_epochs,
    augment=True, shuffle=True, seed=random_seed)

val_hands_dataset = get_hands_dataset_for_superres(
    phase='test', scale_factor=scale_factor, batch_size=batch_size, num_epochs=1,
    augment=False, shuffle=False, seed=random_seed)

# Let us make sure our pipelines are working as expected, by visualizing a batch. For a one-on-one comparison and to better measure the effect of the downsampling, we use TensorFlow to re-scale the images back to their original dimensions.
from plot_utils import plot_image_grid

val_hands_dataset_show = val_hands_dataset.take(1)
val_images_input, val_images_target = next(val_hands_dataset_show.__iter__())
val_images_input = val_images_input[num_show:(num_show * 2)]  # skipping 1st "num_show" batch
val_images_target = val_images_target[num_show:(num_show * 2)]

# Resizing the image back with default method to show the artifacts it causes:
val_images_input_resized = tf.image.resize(val_images_input, tf.shape(val_images_target)[1:3])
val_psnr_result = tf.image.psnr(val_images_target, val_images_input_resized, max_val=1.)

# Displaying some examples:
figure = plot_image_grid([val_images_input_resized.numpy(), val_images_target.numpy()],
                         titles=["scaled", "original"], transpose=True)
figure.show()

print("PSNR for each pair: {}".format(val_psnr_result.numpy()))
# We can clearly see the upscaling artifacts / missing details in the tampered images, as confirmed by their low PSNR. For later comparison, let us compute the average PSNR for the whole validation dataset:

psnr_val = tf.convert_to_tensor([], dtype=tf.float32)
for v_images_input, v_images_target in val_hands_dataset:
    v_images_input = tf.image.resize(v_images_input, tf.shape(v_images_target)[1:3])
    val_psnr_result = tf.image.psnr(v_images_target, v_images_input, max_val=1.)

    psnr_val = tf.concat((psnr_val, val_psnr_result), axis=0)

num_images = psnr_val.shape[0]
psnr_val = tf.reduce_mean(psnr_val).numpy()
print("Avg PSNR using default `tf.image.resize_images()` to scale up the {} val images: {}".format(
    num_images, psnr_val))

# Now
# Building and Training a Deep Auto-Encoder
# Convolutional Auto-Encoder
