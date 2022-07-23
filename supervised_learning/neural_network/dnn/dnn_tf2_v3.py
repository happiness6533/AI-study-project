import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 데이터
# mnist 데이터
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

# 2. 모델 생성
input = tf.keras.Input(shape=(28, 28))
input_flatten = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
hidden1 = tf.keras.layers.Dense(units=16, activation='relu')(input_flatten)
hidden2 = tf.keras.layers.Dense(units=16, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(units=16, activation='relu')(hidden1 + hidden2)
output = tf.keras.layers.Dense(units=10, activation='softmax')(hidden3)
model = tf.keras.Model(input, output)

# 3. 모델 트레이닝
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)

# 4. 모델 평가
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

# 5. 모델 배포
