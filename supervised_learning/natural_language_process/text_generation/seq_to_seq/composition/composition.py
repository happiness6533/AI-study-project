from __future__ import print_function
import sys
from music21 import *
import numpy as np
from compositionProject.grammar import *
from compositionProject.qa import *
from compositionProject.preprocess import *
from compositionProject.music_utils import *
from compositionProject.data_utils import *
from keras.models import load_model, Model
import keras.layers as kl
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

# 데이터
# x : (m, 시간 t, 노트 78)
# y : (시간 t, m, 노트 78) / x보다 1 step 밀려있다
# n_values : 78
# indices_values : 사전 : 키 = 정수, 값 = 노트
X, Y, n_values, indices_values = load_music_utils()

# lstmsingle cell 정의

# rnn 셀 하나 내부의 액티베이션 노드 수 = 64
n_a = 64
# output 모양조절 객체
reshapor = Reshape((1, 78))
# rnn cell 생성기 객체
# return_state: 마지막 리턴값을 아웃풋에 추가한다 안한다 결정하는 불린
LSTM_cell = LSTM(n_a, return_state=True)
# fc 생성기 객체(아웃풋 차원수, 액티베이션)
densor = Dense(n_values, activation='softmax')


# rnn 모델
def djmodel(Tx, n_a, n_values):
    # 예약 인풋, a0, c0 생성
    X = Input(shape=(Tx, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    outputs = []

    for t in range(Tx):
        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:, t, :])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)

    # 모델 객체 생성
    model = Model(inputs=[X, a0, c0], outputs=outputs, name="model")

    return model


# 모델 생성 및 최적화 컴파일
model = djmodel(Tx=30, n_a=64, n_values=78)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# lstm모델을 사용한 샘플링 모델
def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = Lambda(one_hot)(out)

    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs, name="inference_model")

    return inference_model


# 샘플링 모델 실행 및 결과 얻기
inference_model = music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50)
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

# 트레이닝
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
model.fit([X, a0, c0], list(Y), epochs=100)

# 생성
out_stream = generate_music(inference_model)
