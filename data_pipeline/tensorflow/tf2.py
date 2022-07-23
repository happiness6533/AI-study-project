import tensorflow as tf

# 텐서플로우2
# 1. 텐서플로우는 미분을 도와준다
# 2. 즉시 실행과 컴파일의 2가지 모드가 있다
# 3. 케라스를 딥러닝의 대장군으로 사용한다
# 우선 대부분의 모델은 케라스를 사용해서 빌드한다
# 하위 api는 커스텀할때 사용한다
# tf.keras.modules.variables(), tf.keras.modules.trainable_variables()
# tf.keras.layers > Dense, Conv, Conv2d, RNN, LSTM / __call__ build() add_weight() add_loss() >  call()
# tf.keras.Model > compile, fit, evaluate
# tf.keras.Sequential
# tf.keras.network > layers, summary, save

# 상수
# c = tf.constant([[1., 2.],
#                  [3., 4.]])
# print(c.shape)
# print(c.dtype)
# print(tf.zeros(shape=(2, 2)))
# print(tf.ones(shape=(2, 2)))
# print(tf.random.normal(shape=(2, 2), mean=0., stddev=1.))
# print(tf.random.uniform(shape=(2, 2), minval=0, maxval=10))

# 미지수
# x = tf.Variable(tf.random.normal(shape=(2, 2), mean=0., stddev=1.))
# print(x)
# x.assign(tf.zeros(shape=(2, 2)))
# x.assign_add(tf.ones(shape=(2, 2)))
# x.assign_sub(tf.ones(shape=(2, 2)))
# x.assign_add(tf.ones(shape=(2, 2)))
# print(x)

# 미분계수 구하기1
# c1 = tf.random.uniform(shape=(2, 2), minval=1, maxval=4, dtype='float32')
# with tf.GradientTape() as tape:
#     tape.watch(c1)
#     c2 = tf.random.uniform(shape=(2, 2), minval=1, maxval=4, dtype='float32')
#     c3 = tf.math.square(c1) + tf.math.square(c2)
# dc3_dc1 = tape.gradient(c3, c1)
# print(dc3_dc1)

# # 미분계수 구하기2(변수는 워치를 자동으로 실행)
# with tf.GradientTape() as tape:
#     y = tf.square(x) + c
# dy_dx = tape.gradient(y, x)
# print(dy_dx)

# 2차 미분계수를 구하세요
x = tf.Variable(tf.random.normal(shape=(2, 2), mean=0., stddev=1.))
c = tf.constant([[1., 2.], [3., 4.]])
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = tf.math.square(x) + x + c
dy_dx = inner_tape.gradient(y, x)
d2y_dx2 = outer_tape.gradient(dy_dx, x)
print(dy_dx)
print(d2y_dx2)



import tensorflow as tf

num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
callbacks = [tf.keras.callbacks.TensorBoard('./keras')]
model.fit(x_train, y_train, epochs=25, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)