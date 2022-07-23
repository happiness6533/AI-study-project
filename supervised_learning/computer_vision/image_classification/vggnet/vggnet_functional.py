import tensorflow as tf


def VGGNET(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=input_shape,
                               kernel_size=(3, 3),
                               filters=32,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])

    return model


def VGGNET16(input_shape, num_classes):
    return VGGNET(input_shape, num_classes)


def VGGNET32():
    pass
