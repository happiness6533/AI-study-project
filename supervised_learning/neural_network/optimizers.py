import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


def lenet5(name, num_classes):
    model = Sequential(name=name)
    model.add(Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\n\033[94m', '\033[92m'
    log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
    log_end_format = '\033[0m'

    optimizers_examples = {
        'sgd': tf.keras.optimizers.SGD(),
        'momentum': tf.keras.optimizers.SGD(momentum=0.9),
        'nag': tf.keras.optimizers.SGD(momentum=0.9, nesterov=True),
        'adagrad': tf.keras.optimizers.Adagrad(),
        'adadelta': tf.keras.optimizers.Adadelta(),
        'rmsprop': tf.keras.optimizers.RMSprop(),
        'adam': tf.keras.optimizers.Adam()
    }

    history_per_optimizer = dict()
    print("Experiment: {0}start{1} (training logs = off)".format(log_begin_red, log_end_format))

    num_classes = 10
    img_rows, img_cols, img_ch = 28, 28, 1
    input_shape = (img_rows, img_cols, img_ch)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    print('Training data: {}'.format(x_train.shape))
    print('Testing data: {}'.format(x_test.shape))

    for optimizer_name in optimizers_examples:
        tf.random.set_seed(42)
        np.random.seed(42)

        model = lenet5("lenet_{}".format(optimizer_name), num_classes)
        optimizer = optimizers_examples[optimizer_name]
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(f"Training with {optimizer_name}: {log_begin_red}start{log_end_format}")
        history = model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_test, y_test), verbose=1)
        history_per_optimizer[optimizer_name] = history
        print(f"Training with {optimizer_name}: {log_begin_green}done{log_end_format}.")
        print()

    print(f"Experiment: {log_begin_green}done{log_end_format}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='col')
    ax[0, 0].set_title("loss")
    ax[0, 1].set_title("val-loss")
    ax[1, 0].set_title("accuracy")
    ax[1, 1].set_title("val-accuracy")

    lines, labels = [], []
    for optimizer_name in history_per_optimizer:
        history = history_per_optimizer[optimizer_name]
        ax[0, 0].plot(history.history['loss'])
        ax[0, 1].plot(history.history['val_loss'])
        ax[1, 0].plot(history.history['accuracy'])
        line = ax[1, 1].plot(history.history['val_accuracy'])
        lines.append(line[0])
        labels.append(optimizer_name)

    fig.legend(lines, labels, loc='center right', borderaxespad=0.1)
    plt.subplots_adjust(right=0.85)
    plt.show()
