from keras import backend as K
import numpy as np
import tensorflow as tf

K.set_image_data_format('channels_first')

np.set_printoptions(threshold=np.nan)


class VggLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(VggLayer, self).__init__()
        x1 = tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, padding='same',
                                    activation='relu'),
        x2 = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
        x3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        x4 = tf.keras.layers.Dropout(rate=0.5),
        x5 = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),
        x6 = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),
        x7 = tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        x8 = tf.keras.layers.Dropout(rate=0.5),
        x10 = tf.keras.layers.Flatten(),
        x11 = tf.keras.layers.Dense(units=512, activation='relu'),
        x12 = tf.keras.layers.Dropout(rate=0.5),
        x13 = tf.keras.layers.Dense(units=256, activation='relu'),
        x14 = tf.keras.layers.Dropout(rate=0.5),
        x15 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, input):
        output = self.x1(input)
        output = self.x2(output)
        output = self.x3(output)
        output = self.x4(output)
        output = self.x5(output)
        output = self.x6(output)
        output = self.x7(output)
        output = self.x8(output)
        output = self.x9(output)
        output = self.x10(output)
        output = self.x11(output)
        output = self.x12(output)
        output = self.x13(output)
        output = self.x14(output)
        output = self.x15(output)
        return output


class Siamese(tf.keras.models.Model):
    def __init__(self):
        super(Siamese, self).__init__()
        vgg = VggLayer()

    def call(self, inputs, y):
        o1 = vgg(input[1])
        o2 = vgg(input[2])

        return o1, o2


model = Siamese()


def triplet_loss(y_true, y_pred):
    """
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    alpha = 0.2
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1, keep_dims=False)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1, keep_dims=False)
    basic_loss = pos_dist + alpha - neg_dist
    loss = tf.reduce_sum(tf.maximum(0.0, basic_loss), keep_dims=False)

    return loss


model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
model.summary()


# 이미지 인코딩
def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


# 신분 증명
def who_are_you(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open


# 감정 판단
def how_are_you(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##
    min_dist = 100
    for name in database.keys():
        dist = np.linalg.norm(encoding - database[name])
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity
