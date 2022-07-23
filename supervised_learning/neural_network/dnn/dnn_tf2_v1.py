import tensorflow as tf
import matplotlib.pyplot as plt

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

# 2. 모델
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 3. 모델 로스/옵티마이저/메트릭
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델의 다양한 정보들
# model.inputs
# model.outputs
# model.summary()
# model.save() >> tf.keras.models.load_model()을 사용해서 저장된 파일을 인스턴스로 불러올 수 있다
# model.save_weights()
# model.load_weights()
# model.get_weights()
# model.set_weights()

# 4. 모델 트레이닝
# 콜백을 추가해야 함
# 기본 콜백들 = CSVLogger, EarlyStopping(과적합을 피하기 위해 적당히 중지),
# LearningRateScheduler(스케줄에 따라 학습률 변경), ReduceLROnPlateau(손실과 메트릭 개선이 없는 경우 학습률 자동 감소)
# tf.keras.callbacks.Callback을 쓰면 맞춤형 서브클래스콜백을 만들 수 잇다
model.fit(x_train, y_train, batch_size=100, epochs=10)

# 5. 모델 평가
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

# 6. 모델 배포
