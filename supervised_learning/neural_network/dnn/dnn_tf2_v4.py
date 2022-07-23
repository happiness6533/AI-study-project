import matplotlib.pyplot as plt
import tensorflow as tf

# 1. 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# 2. 레이어
class CustomDense1(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomDense1, self).__init__()
        self.w = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0., stddev=1.),
                             trainable=True)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(units,)),
                             trainable=True)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


custom_dense1 = CustomDense1(input_dim=2, units=1, activation='relu')
assert custom_dense1.weights == [custom_dense1.w, custom_dense1.b]
assert custom_dense1.trainable_weights == [custom_dense1.w, custom_dense1.b]
assert custom_dense1.non_trainable_weights == []


# 레이어 생성2(build 메소드)
class CustomDense2(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomDense2, self).__init__()
        self.input_dim = input_dim
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    # 모델 관련된 것들은 여기에 정의하자
    # 그냥 레이어를 바로 정의해도 큰 문제 없음
    def build(self):
        self.w = self.add_weight(shape=(self.input_dim, self.units), initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


# 레이어 생성3(call 메소드 training)
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(CustomDropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


# 레이어 생성4(레이어 조합)
class CustomCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomCombinationLayer, self).__init__()
        self.custom_dense1 = CustomDense1(input_dim, units, activation)
        self.custom_dropout = CustomDropout(0.5)

    def call(self, inputs, training=None):
        a = self.custom_dense1(inputs)
        return self.custom_dropout(a, training=training)


# 3. 모델
class CustomModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.hidden_layer1 = CustomCombinationLayer(input_dim=784, units=16, activation='relu')
        self.hidden_layer2 = CustomCombinationLayer(input_dim=16, units=16, activation='relu')
        self.hidden_layer3 = CustomCombinationLayer(input_dim=16, units=16, activation='relu')
        self.output_layer = CustomCombinationLayer(input_dim=16, units=10, activation='softmax')

    def call(self, inputs, training=True):
        hidden1 = self.hidden_layer1(inputs, training=training)
        hidden2 = self.hidden_layer2(hidden1, training=training)
        hidden3 = self.hidden_layer3(hidden2, training=training)
        return self.output_layer(hidden3, training=training)


model = CustomModel()

# 4. 모델 로스/옵티마이저/메트릭
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()
accuracy = tf.keras.metrics.CategoricalAccuracy()


# 5. 모델 트레이닝/평가
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss(y, logits)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy.update_state(y, logits)

    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = model(x, training=False)
    accuracy.update_state(y, logits)


for epoch in range(5):
    print('%d번째 epoch' % (epoch + 1))
    for index, (x, y) in enumerate(train_data):
        loss_value = train_step(x, y)
        if index % 100 == 0:
            print('%d 단계 / loss_value: %f / accuracy: %f' % (index, float(loss_value), float(accuracy.result())))
    for step, (x, y) in enumerate(test_data):
        test_step(x, y)
        print('test accuracy: %f' % (float(accuracy.result())))

# 6. 모델 배포
