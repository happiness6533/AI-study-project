if __name__ == "__main__":
    import os
    import math
    import glob
    import collections
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    import cifar_utils
    from keras_custom_callbacks import SimpleLogCallback
    from classification_utils import load_image, process_predictions, display_predictions

    # 데이터
    cifar_info = cifar_utils.get_info()
    print(cifar_info)
    print(cifar_info.features["label"].names)
    print(cifar_info.features["coarse_label"].names)
    num_classes = cifar_info.features['label'].num_classes
    num_train_imgs = cifar_info.splits['train'].num_examples
    num_val_imgs = cifar_info.splits['test'].num_examples
    input_shape = [224, 224, 3]
    batch_size = 32
    num_epochs = 300
    random_seed = 42
    train_cifar_dataset = cifar_utils.get_dataset(phase='train', batch_size=batch_size, num_epochs=num_epochs,
                                                  shuffle=True, input_shape=input_shape, seed=random_seed)
    val_cifar_dataset = cifar_utils.get_dataset(phase='test', batch_size=batch_size, num_epochs=1, shuffle=False,
                                                input_shape=input_shape, seed=random_seed)
    print('Training dataset instance: {}'.format(train_cifar_dataset))

    # 2. 모델 생성
    train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
    val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_shape=input_shape,
                                                       classes=num_classes)
    resnet50.summary()

    accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='acc')
    top5_accuracy_metric = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
    resnet50.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=[accuracy_metric, top5_accuracy_metric])

    metrics_to_print = collections.OrderedDict([("loss", "loss"),
                                                ("v-loss", "val_loss"),
                                                ("acc", "acc"),
                                                ("v-acc", "val_acc"),
                                                ("top5-acc", "top5_acc"),
                                                ("v-top5-acc", "val_top5_acc")])

    model_dir = './models/resnet_keras_app'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc', restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
                                           period=5, monitor='val_loss'),
        SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, log_frequency=1)
    ]

    # 3. 모델 최적화
    history = resnet50.fit(train_cifar_dataset, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
                           validation_data=val_cifar_dataset, validation_steps=val_steps_per_epoch,
                           verbose=0, callbacks=callbacks)

    # 4. 모델 평가
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

    test_filenames = glob.glob(os.path.join('res', '*'))
    test_images = np.asarray([load_image(file, size=input_shape[:2]) for file in test_filenames])
    print('Test Images: {}'.format(test_images.shape))

    image_batch = test_images[:16]
    cifar_original_image_size = cifar_info.features['image'].shape[:2]
    image_batch_low_quality = tf.image.resize(image_batch, cifar_original_image_size)
    image_batch_low_quality = tf.image.resize(image_batch_low_quality, input_shape[:2])

    predictions = resnet50.predict_on_batch(image_batch_low_quality)

    class_readable_labels = cifar_info.features["label"].names
    top5_labels, top5_probabilities = process_predictions(predictions.numpy(), class_readable_labels)

    display_predictions(image_batch, top5_labels, top5_probabilities)
