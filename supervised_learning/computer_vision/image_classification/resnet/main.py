if __name__ == "__main__":
    import os
    import math
    import glob
    import collections
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # from cifar_utils import get_dataset, CIFAR_BUILDER, get_info
    from resnet_functional import ResNet50, ResNet18
    from keras_custom_callbacks import SimpleLogCallback
    from classification_utils import process_predictions, display_predictions, load_image

    # 1. 데이터
    # cifar_info = get_info()
    # print(cifar_info)
    # print(cifar_info.features["label"].names)
    # print(cifar_info.features["coarse_label"].names)
    # num_classes = CIFAR_BUILDER.info.features['label'].num_classes
    # num_train_imgs = CIFAR_BUILDER.info.splits['train'].num_examples
    # num_val_imgs = CIFAR_BUILDER.info.splits['test'].num_examples
    # input_shape = [224, 224, 3]
    # batch_size = 32
    # num_epochs = 300

    # train_cifar_dataset = get_dataset(phase='train', num_epochs=300, input_shape=[224, 224, 3], batch_size=32)
    # val_cifar_dataset = get_dataset(phase='test', input_shape=[224, 224, 3], batch_size=32)
    # print('Training dataset instance: {}'.format(train_cifar_dataset))
    #
    # # 2. 모델 생성
    # train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
    # val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)
    #
    # resnet50 = ResNet50(input_shape=input_shape, num_classes=num_classes)
    # resnet50.summary()
    #
    # accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='acc')
    # top5_accuracy_metric = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
    # resnet50.compile(optimizer='adam',
    #                  loss='sparse_categorical_crossentropy',
    #                  metrics=[accuracy_metric, top5_accuracy_metric])
    #
    # metrics_to_print = collections.OrderedDict([("loss", "loss"),
    #                                             ("v-loss", "val_loss"),
    #                                             ("acc", "acc"),
    #                                             ("v-acc", "val_acc"),
    #                                             ("top5-acc", "top5_acc"),
    #                                             ("v-top5-acc", "val_top5_acc")])
    # model_dir = './models/resnet_from_scratch'
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc', restore_best_weights=True),
    #     tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
    #     # save
    #     tf.keras.callbacks.ModelCheckpoint(
    #         os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), period=5
    #     ),
    #     # custom callback
    #     SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=2)
    # ]
    #
    # # 3. 모델 최적화
    # history = resnet50.fit(train_cifar_dataset,
    #                        epochs=num_epochs,
    #                        steps_per_epoch=train_steps_per_epoch,
    #                        validation_data=val_cifar_dataset,
    #                        validation_steps=val_steps_per_epoch,
    #                        verbose=0,
    #                        callbacks=callbacks)
    #
    # # 4. 모델 평가
    # fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col')
    # ax[0, 0].set_title("loss")
    # ax[0, 1].set_title("val-loss")
    # ax[1, 0].set_title("acc")
    # ax[1, 1].set_title("val-acc")
    # ax[2, 0].set_title("top5-acc")
    # ax[2, 1].set_title("val-top5-acc")
    #
    # ax[0, 0].plot(history.history['loss'])
    # ax[0, 1].plot(history.history['val_loss'])
    # ax[1, 0].plot(history.history['acc'])
    # ax[1, 1].plot(history.history['val_acc'])
    # ax[2, 0].plot(history.history['top5_acc'])
    # ax[2, 1].plot(history.history['val_top5_acc'])
    #
    # best_val_acc = max(history.history['val_acc']) * 100
    # best_val_top5 = max(history.history['val_top5_acc']) * 100
    #
    # print('Best val acc:  {:2.2f}%'.format(best_val_acc))
    # print('Best val top5: {:2.2f}%'.format(best_val_top5))
    #
    # test_filenames = glob.glob(os.path.join('res', '*'))
    # test_images = np.asarray([load_image(file, size=input_shape[:2]) for file in test_filenames])
    # print('Test Images: {}'.format(test_images.shape))
    #
    # image_batch = test_images[:16]
    # cifar_original_image_size = CIFAR_BUILDER.info.features['image'].shape[:2]
    # image_batch_low_quality = tf.image.resize(image_batch, cifar_original_image_size)
    # image_batch_low_quality = tf.image.resize(image_batch_low_quality, input_shape[:2])
    #
    # predictions = resnet50.predict_on_batch(image_batch_low_quality)
    # print('Predicted class probabilities: {}'.format(predictions.shape))
    #
    # class_readable_labels = CIFAR_BUILDER.info.features["label"].names
    # top5_labels, top5_probabilities = process_predictions(predictions, class_readable_labels)
    #
    # display_predictions(image_batch, top5_labels, top5_probabilities)

    # 데이터 로드 train / val / test
    test_data_dir = "C://Users//Happiness//developer//study-ai//data//bird//test"
    train_data_dir = "C://Users//Happiness//developer//study-ai//data//bird//train"

    batch_size = 32
    img_height = 224
    img_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        labels="inferred",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        labels="inferred",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    # 메모리 캐시화
    # AUTOTUNE = tf.data.AUTOTUNE
    #
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 시각화
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    # 데이터 정규화
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # 상수
    num_classes = len(class_names)
    num_train_imgs = len(normalized_train_ds)
    num_test_imgs = len(normalized_test_ds)
    num_val_imgs = len(normalized_val_ds)
    input_shape = [224, 224, 3]
    batch_size = 32
    num_epochs = 300

    # 2. 모델 생성
    train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
    test_steps_per_epoch = math.ceil(num_test_imgs / batch_size)
    val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

    resnet18 = ResNet18(input_shape=input_shape, num_classes=num_classes)
    resnet18.summary()

    accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='acc')
    top5_accuracy_metric = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
    resnet18.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=[accuracy_metric, top5_accuracy_metric])

    metrics_to_print = collections.OrderedDict([("loss", "loss"),
                                                ("v-loss", "val_loss"),
                                                ("acc", "acc"),
                                                ("v-acc", "val_acc"),
                                                ("top5-acc", "top5_acc"),
                                                ("v-top5-acc", "val_top5_acc")])
    model_dir = './models/resnet_from_scratch'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc', restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
        # save
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), period=5
        ),
        # custom callback
        SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=2)
    ]

    # 3. 모델 최적화
    history = resnet18.fit(normalized_train_ds,
                           epochs=num_epochs,
                           steps_per_epoch=train_steps_per_epoch,
                           validation_data=normalized_val_ds,
                           validation_steps=val_steps_per_epoch,
                           verbose=1,
                           callbacks=callbacks)

    fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col')
    ax[0, 0].set_title("loss")
    ax[0, 1].set_title("val-loss")
    ax[1, 0].set_title("acc")
    ax[1, 1].set_title("val-acc")
    ax[2, 0].set_title("top5-acc")
    ax[2, 1].set_title("val-top5-acc")

    ax[0, 0].plot(history.history['loss'])
    ax[0, 1].plot(history.history['val_loss'])
    ax[1, 0].plot(history.history['acc'])
    ax[1, 1].plot(history.history['val_acc'])
    ax[2, 0].plot(history.history['top5_acc'])
    ax[2, 1].plot(history.history['val_top5_acc'])

    best_val_acc = max(history.history['val_acc']) * 100
    best_val_top5 = max(history.history['val_top5_acc']) * 100

    print('Best val acc:  {:2.2f}%'.format(best_val_acc))
    print('Best val top5: {:2.2f}%'.format(best_val_top5))
    resnet18.predict()

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    import pandas as pd
    import numpy as np
    import random
    import os
    import math
    import collections
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import (Input,
                                         Activation,
                                         Dense,
                                         Flatten,
                                         Conv2D,
                                         MaxPooling2D,
                                         AveragePooling2D,
                                         BatchNormalization,
                                         add,
                                         Dropout)


    def _res_conv(filters, kernel_size=3, padding='same', strides=1, use_relu=True, use_bias=False, name='cbr',
                  kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)):
        def layer_fn(x):
            conv = Conv2D(
                filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_c')(x)
            res = BatchNormalization(axis=-1, name=name + '_bn')(conv)
            if use_relu:
                res = Activation("relu", name=name + '_r')(res)
            return res

        return layer_fn


    def _merge_with_shortcut(kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='block'):
        def layer_fn(x, x_residual):
            x_shape = tf.keras.backend.int_shape(x)
            x_residual_shape = tf.keras.backend.int_shape(x_residual)
            if x_shape == x_residual_shape:
                shortcut = x
            else:
                strides = (
                    int(round(x_shape[1] / x_residual_shape[1])),  # vertical stride
                    int(round(x_shape[2] / x_residual_shape[2]))  # horizontal stride
                )
                x_residual_channels = x_residual_shape[3]
                shortcut = Conv2D(
                    filters=x_residual_channels, kernel_size=(1, 1), padding="valid", strides=strides,
                    kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                    name=name + '_shortcut_c')(x)

            merge = add([shortcut, x_residual])
            return merge

        return layer_fn


    def _residual_block_basic(filters, kernel_size=3, strides=1, use_bias=False, name='res_basic',
                              kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)):
        def layer_fn(x):
            x_conv1 = _res_conv(
                filters=filters, kernel_size=kernel_size, padding='same', strides=strides,
                use_relu=True, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_cbr_1')(x)
            x_residual = _res_conv(
                filters=filters, kernel_size=kernel_size, padding='same', strides=1,
                use_relu=False, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_cbr_2')(x_conv1)
            merge = _merge_with_shortcut(kernel_initializer, kernel_regularizer, name=name)(x, x_residual)
            merge = Activation('relu')(merge)
            return merge

        return layer_fn


    def _residual_block_bottleneck(filters, kernel_size=3, strides=1, use_bias=False, name='res_bottleneck',
                                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)):
        def layer_fn(x):
            x_bottleneck = _res_conv(
                filters=filters, kernel_size=1, padding='valid', strides=strides,
                use_relu=True, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_cbr1')(x)
            x_conv = _res_conv(
                filters=filters, kernel_size=kernel_size, padding='same', strides=1,
                use_relu=True, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_cbr2')(x_bottleneck)
            x_residual = _res_conv(
                filters=filters * 4, kernel_size=1, padding='valid', strides=1,
                use_relu=False, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                name=name + '_cbr3')(x_conv)
            merge = _merge_with_shortcut(kernel_initializer, kernel_regularizer, name=name)(x, x_residual)
            merge = Activation('relu')(merge)
            return merge

        return layer_fn


    def _residual_macroblock(block_fn, filters, repetitions=3, kernel_size=3, strides_1st_block=1, use_bias=False,
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='res_macroblock'):
        def layer_fn(x):
            for i in range(repetitions):
                block_name = "{}_{}".format(name, i)
                strides = strides_1st_block if i == 0 else 1
                x = block_fn(filters=filters, kernel_size=kernel_size,
                             strides=strides, use_bias=use_bias,
                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                             name=block_name)(x)
            return x

        return layer_fn


    def ResNet(input_shape, repetitions, num_classes=1000, block_fn=_residual_block_basic, use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)):
        inputs = Input(shape=input_shape)
        conv = _res_conv(
            filters=64, kernel_size=7, strides=2, use_relu=True, use_bias=use_bias,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inputs)
        maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv)

        filters = 64
        strides = 2
        res_block = maxpool
        for i, repet in enumerate(repetitions):
            block_strides = strides if i != 0 else 1
            macroblock_name = "block_{}".format(i)
            res_block = _residual_macroblock(
                block_fn=block_fn, repetitions=repet, name=macroblock_name,
                filters=filters, strides_1st_block=block_strides, use_bias=use_bias,
                kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(res_block)
            if i == 0:
                filters = min(int(filters * 2), 1024)  # we limit to 1024 filters max
            if i == 1:
                filters = min(int(filters * 1.5), 1024)  # we limit to 1024 filters max
            if i == 2:
                filters = min(int(filters * 1.5), 1024)  # we limit to 1024 filters max
            if i == 3:
                filters = min(int(filters * 1.5), 1024)  # we limit to 1024 filters max

        res_spatial_dim = tf.keras.backend.int_shape(res_block)[1:3]
        avg_pool = AveragePooling2D(pool_size=res_spatial_dim, strides=1)(res_block)
        flatten = Flatten()(avg_pool)
        predictions = Dense(units=num_classes, kernel_initializer=kernel_initializer,
                            activation='softmax')(flatten)
        model = Model(inputs=inputs, outputs=predictions)

        return model


    def ResNet_Custom(input_shape, num_classes, use_bias=True, kernel_initializer='he_normal', kernel_regularizer=None):
        return ResNet(input_shape, repetitions=(2, 2, 2, 2), num_classes=num_classes, block_fn=_residual_block_basic,
                      use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)


    def converto_to_indices(x):
        return (x[0], tf.argmax(x[1], axis=1))


    def prepare_data(train_data_dir, test_data_dir, batch_size, input_shape):
        # 상수
        [img_height, img_width, _] = input_shape

        # 데이터 로드
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_data_dir,
            labels="inferred",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_data_dir,
            labels="inferred",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = train_ds.class_names

        # 정규화
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        return train_ds, val_ds, class_names


    def prepare_augmented_train_data(train_data_dir, batch_size, input_shape):
        [img_height, img_width, _] = input_shape

        # 데이터 증강
        train_img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.20,
            height_shift_range=0.20,
            horizontal_flip=True,
            rescale=1. / 255,
            validation_split=0.2,
        )
        train_generator = map(converto_to_indices,
                              train_img_gen.flow_from_directory(train_data_dir, subset="training",
                                                                target_size=(img_height, img_width),
                                                                seed=123, batch_size=batch_size))
        train_ds_generator = tf.data.Dataset.from_generator(lambda: train_generator,
                                                            output_types=(tf.float32, tf.float32))
        return train_ds_generator


    def train_bird_classifier(model_dir, data_augment, train_data_dir, test_data_dir, num_classes, batch_size,
                              num_epochs,
                              is_first, recover_file_name, save_file_name):
        # 데이터
        input_shape = [224, 224, 3]
        train_ds, val_ds, class_names = prepare_data(train_data_dir, test_data_dir, batch_size, input_shape)
        if data_augment:
            train_ds_generator = prepare_augmented_train_data(train_data_dir, batch_size, input_shape)
        num_train_imgs = len(train_ds) * batch_size

        # 모델
        resnet_custom = ResNet_Custom(input_shape=input_shape, num_classes=num_classes)
        resnet_custom.summary()
        accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='acc')
        resnet_custom.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=[accuracy_metric])
        if data_augment:
            monitor = 'val_acc'
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(model_dir, save_file_name),
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=True,
                ),
            ]
        else:
            monitor = 'acc'
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(model_dir, save_file_name),
                    monitor=monitor,
                    save_best_only=True,
                    save_weights_only=True,
                ),
            ]

        if data_augment:
            resnet_custom.load_weights(os.path.join(model_dir, recover_file_name))
            history = resnet_custom.fit(train_ds_generator,
                                        epochs=num_epochs,
                                        steps_per_epoch=min(math.ceil(num_train_imgs / batch_size), 100),
                                        validation_data=val_ds,
                                        verbose=1,
                                        callbacks=callbacks)
        else:
            history = resnet_custom.fit(train_ds,
                                        epochs=num_epochs,
                                        steps_per_epoch=min(math.ceil(num_train_imgs / batch_size), 100),
                                        validation_data=val_ds,
                                        verbose=1,
                                        callbacks=callbacks)


    if __name__ == "__main__":
        import os

        model_dir = '/kaggle/working/models'
        os.mkdir(model_dir)
        model_dir = '/kaggle/working/models/custom_resnet'
        os.mkdir(model_dir)

        train_data_dir = '/kaggle/input/100-bird-species/train'
        test_data_dir = '/kaggle/input/100-bird-species/test'
        batch_size = 64
        num_classes = 400

        is_first = True
        data_augment = False
        num_epochs = 10
        recover_file_name = None
        save_file_name = '10_best_result_weights.h5'
        train_bird_classifier(model_dir, data_augment, train_data_dir, test_data_dir, num_classes, batch_size,
                              num_epochs,
                              is_first, recover_file_name, save_file_name)
        is_first = False
        data_augment = True
        num_epochs = 10
        recover_file_name = '10_best_result_weights.h5'
        save_file_name = '20_best_result_weights.h5'
        train_bird_classifier(model_dir, data_augment, train_data_dir, test_data_dir, num_classes, batch_size,
                              num_epochs,
                              is_first, recover_file_name, save_file_name)

        is_first = False
        data_augment = True
        num_epochs = 10
        recover_file_name = '20_best_result_weights.h5'
        save_file_name = '30_best_result_weights.h5'
        train_bird_classifier(model_dir, data_augment, train_data_dir, test_data_dir, num_classes, batch_size,
                              num_epochs,
                              is_first, recover_file_name, save_file_name)

        is_first = False
        data_augment = True
        num_epochs = 10
        recover_file_name = '30_best_result_weights.h5'
        save_file_name = '40_best_result_weights.h5'
        train_bird_classifier(model_dir, data_augment, train_data_dir, test_data_dir, num_classes, batch_size,
                              num_epochs,
                              is_first, recover_file_name, save_file_name)

        is_first = False
        data_augment = True
        num_epochs = 10
        recover_file_name = '40_best_result_weights.h5'
        save_file_name = '50_best_result_weights.h5'
        train_bird_classifier(model_dir, data_augment, train_data_dir, test_data_dir, num_classes, batch_size,
                              num_epochs,
                              is_first, recover_file_name, save_file_name)

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            '/kaggle/input/100-bird-species/test',
            labels="inferred",
            image_size=(224, 224),
            shuffle=False,
            batch_size=128)
        class_names = test_ds.class_names
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

        resnet_custom = ResNet_Custom(input_shape=[224, 224, 3], num_classes=400)
        accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='acc')
        top5_accuracy_metric = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
        resnet_custom.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=[accuracy_metric, top5_accuracy_metric])
        model_dir = '/kaggle/input/weights'
        resnet_custom.load_weights(os.path.join(model_dir, 'augment_best_result_weights.h5'))
        result = resnet_custom.evaluate(test_ds, batch_size=128, verbose=1)
        print(result)

        test_ds = list(test_ds[1])

        category = []
        for i in order:
            batch_input, batch_label = test_ds[int(i)]
            preds = resnet_custom.predict(batch_input, batch_size=1, verbose=1)
            result = tf.math.argmax(preds[0], output_type=tf.int32).numpy()
            category.append(result)
            print(len(category))
        print(len(category))
        prediction = {'Id': list(range(2000)), 'Category': category}
        prediction_df = pd.DataFrame(prediction, columns=['Id', 'Category'])
        prediction_df.to_csv('/kaggle/working/prediction.csv', index=False)
