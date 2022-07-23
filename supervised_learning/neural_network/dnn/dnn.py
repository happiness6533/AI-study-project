class SimpleNetwork(object):
    def __init__(self,
                 num_inputs, num_outputs,
                 hidden_layers_sizes=(64, 32),
                 activation_function=sigmoid,
                 derivated_activation_function=derivated_sigmoid,
                 loss_function=loss_L2,
                 derivated_loss_function=derivated_loss_L2):
        super().__init__()

        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1], activation_function, derivated_activation_function)
            for i in range(len(layer_sizes) - 1)
        ]

        self.loss_function = loss_function
        self.derivated_loss_function = derivated_loss_function

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x):
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def backward(self, dL_dy):
        for layer in reversed(self.layers):  # from the output layer to the input one
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self, epsilon):
        for layer in self.layers:  # the order doesn't matter here
            layer.optimize(epsilon)

    def evaluate_accuracy(self, X_val, y_val):
        num_corrects = 0
        for i in range(len(X_val)):
            pred_class = self.predict(X_val[i])
            if pred_class == y_val[i]:
                num_corrects += 1
        return num_corrects / len(X_val)

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, num_epochs=5, learning_rate=1e-3):
        num_batches_per_epoch = len(X_train) // batch_size
        do_validation = X_val is not None and y_val is not None
        losses, accuracies = [], []
        for i in range(num_epochs):  # for each training epoch
            epoch_loss = 0
            for b in range(num_batches_per_epoch):  # for each batch composing the dataset
                # Get batch:
                batch_index_begin = b * batch_size
                batch_index_end = batch_index_begin + batch_size
                x = X_train[batch_index_begin: batch_index_end]
                targets = y_train[batch_index_begin: batch_index_end]
                # Optimize on batch:
                predictions = y = self.forward(x)  # forward pass
                L = self.loss_function(predictions, targets)  # loss computation
                dL_dy = self.derivated_loss_function(predictions, targets)  # loss derivation
                self.backward(dL_dy)  # back-propagation pass
                self.optimize(learning_rate)  # optimization of the NN
                epoch_loss += L

            # Logging training loss and validation accuracy, to follow the training:
            epoch_loss /= num_batches_per_epoch
            losses.append(epoch_loss)
            if do_validation:
                accuracy = self.evaluate_accuracy(X_val, y_val)
                accuracies.append(accuracy)
            else:
                accuracy = np.NaN
            print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(i, epoch_loss, accuracy * 100))
        return losses, accuracies


if __name__ == "__main__":
    from supervised_learning.neural_network.math_of_neural_network.math \
        import sigmoid, derivated_sigmoid, derivated_loss_L2, derivated_cross_entropy, cross_entropy
    import numpy as np
    from fcn import FullyConnectedLayer
    import matplotlib
    import matplotlib.pyplot as plt
    import mnist

    np.random.seed(42)
    X_train, y_train = mnist.train_images(), mnist.train_labels()
    X_test, y_test = mnist.test_images(), mnist.test_labels()
    num_classes = 10  # classes are the digits from 0 to 9

    img_idx = np.random.randint(0, X_test.shape[0])
    plt.imshow(X_test[img_idx], cmap=matplotlib.cm.binary)
    plt.axis("off")
    plt.show()

    X_train, X_test = X_train.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)
    print("Pixel values between {} and {}".format(X_train.min(), X_train.max()))

    X_train, X_test = X_train / 255., X_test / 255.
    print("Normalized pixel values between {} and {}".format(X_train.min(), X_train.max()))
    y_train = np.eye(num_classes)[y_train]

    # 훈련
    mnist_classifier = SimpleNetwork(num_inputs=X_train.shape[1],
                                     num_outputs=num_classes, hidden_layers_sizes=[64, 32])
    predictions = mnist_classifier.forward(X_train)  # forward pass
    loss_untrained = mnist_classifier.loss_function(predictions, y_train)  # loss computation

    accuracy_untrained = mnist_classifier.evaluate_accuracy(X_test, y_test)  # Accuracy
    print("Untrained : training loss = {:.6f} | val accuracy = {:.2f}%".format(
        loss_untrained, accuracy_untrained * 100))
    losses, accuracies = mnist_classifier.train(X_train, y_train, X_test, y_test,
                                                batch_size=30, num_epochs=500)
    # note: Reduce the batch size and/or number of epochs if your computer can't
    #       handle the computations / takes too long.
    #       Remember, numpy also uses the CPU, not GPUs as modern Deep Learning
    #       libraries do, hence the lack of computational performance here.

    losses, accuracies = [loss_untrained] + losses, [accuracy_untrained] + accuracies
    fig, ax_loss = plt.subplots()

    color = 'red'
    ax_loss.set_xlim([0, 510])
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Training Loss', color=color)
    ax_loss.plot(losses, color=color)
    ax_loss.tick_params(axis='y', labelcolor=color)

    ax_acc = ax_loss.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'blue'
    ax_acc.set_xlim([0, 510])
    ax_acc.set_ylim([0, 1])
    ax_acc.set_ylabel('Val Accuracy', color=color)
    ax_acc.plot(accuracies, color=color)
    ax_acc.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

    # 테스트
    # We use `np.expand_dims(x, 0)` to simulate a batch (transforming the image shape
    # from (784,) to (1, 784)):
    predicted_class = mnist_classifier.predict(np.expand_dims(X_test[img_idx], 0))
    print('Predicted class: {}; Correct class: {}'.format(predicted_class, y_test[img_idx]))
