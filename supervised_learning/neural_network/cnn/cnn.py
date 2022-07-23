import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from skimage import io  # Package to simply read images

image = io.imread("./res/bird_pic_by_benjamin_planche.png")
print("Image shape: {}".format(image.shape))
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

image = tf.convert_to_tensor(image, tf.float32, name="input_image")
image = tf.expand_dims(image, axis=0)  # we expand our tensor, adding a dimension at position 0 for   (𝐵,𝐻,𝑊,𝐷)
image = tf.expand_dims(image,
                       axis=-1)  # we expand our tensor, adding a dimension at position 0 for as our image is grayscale and have only one channel, it currently doesn't explicitely have a 4th depth dimension
print("Tensor shape: {}".format(image.shape))

# 가우시안 블러
kernel = tf.constant([[1 / 16, 2 / 16, 1 / 16],
                      [2 / 16, 4 / 16, 2 / 16],
                      [1 / 16, 2 / 16, 1 / 16]], tf.float32, name="gaussian_kernel")
# the convolution method requires the filter tensor to be of shape  (𝑘,𝑘,𝐷,𝑁) where d = n = 1
kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
blurred_image = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")
blurred_res = blurred_image.numpy()
# We "unbatch" our result by selecting the first (and only) image; we also remove the depth dimension:
blurred_res = blurred_res[0, ..., 0]

plt.imshow(blurred_res, cmap=plt.cm.gray)

# 컨투어 커널
kernel = tf.constant([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]], tf.float32, name="edge_kernel")
kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
edge_image = tf.nn.conv2d(image, kernel, strides=[1, 2, 2, 1], padding="SAME")
edge_res = edge_image.numpy()[0, ..., 0]
plt.imshow(edge_res, cmap=plt.cm.gray)
# If you look closely, the image has a white border.
# This is caused by the zero-padding (since we chose padding "SAME"),
# detected as a contour by the kernel.
# Indeed, it disappears if we don't pad the image:
edge_image = tf.nn.conv2d(image, kernel, strides=[1, 2, 2, 1], padding="VALID")
edge_res = edge_image.numpy()[0, ..., 0]
plt.imshow(edge_res, cmap=plt.cm.gray)

# 풀링
# tf.keras.layers.AvgPool2D가 역시 더 상위 API로 존재함
avg_pooled_image = tf.nn.avg_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
avg_res = avg_pooled_image.numpy()[0, ..., 0]
plt.imshow(avg_res, cmap=plt.cm.gray)

# tf.keras.layers.MaxPool2D가 역시 더 상위 API로 존재함
max_pooled_image = tf.nn.max_pool(image, ksize=[1, 10, 10, 1], strides=[1, 2, 2, 1], padding="SAME")
max_res = max_pooled_image.numpy()[0, ..., 0]
plt.imshow(max_res, cmap=plt.cm.gray)
