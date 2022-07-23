from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from translationProject.nmt_utils import *
import matplotlib.pyplot as plt

# 데이터
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

dataset[:10]
print(human_vocab)
print(machine_vocab)

# 데이터 전처리
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])

# Defined shared projects as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, activation="relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


# 어텐션lstm 총정리
# 1. 어텐션 이전 레이어인 bi-rnn(길이 : input길이)을 작동시켜서 모든 스텝 t에 대한 a(t)를 뽑아낸다
# 2. a(t)에 s(t-1)을 복제해서 이어붙이고 적당한 덴스레이어를 거친 다음 소프트맥스를 취하면 알파를 얻는다
# 3. 알파와 a(t)의 내적합을 계산해서 context(t)를 얻는다
# 4. context(t)와 s(t-1)과 메모리셀 c(t-1)을 넣고 어텐션lstm(길이 : output길이)을 가동시킨다
# 5. 매 스텝마다 나오는 결론에 소프트맥스를 취해서 결과를 얻는다

# 아래 함수에 대한 설명
# 2. a(t)에 s(t-1)을 복제해서 이어붙이고 적당한 덴스레이어를 거친 다음 소프트맥스를 취하면 알파를 얻는다
# 3. 알파와 a(t)의 내적합을 계산해서 context(t)를 얻는다
def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    # 포스트 어텐션 lstm의 이 전 상태값을 여러번 반복해서 사용해야 하기 때문에, 반복횟수만큼 복제된 새로운 텐서를 생성하자
    # 반복자를 거치면 : (m, n) >> (m, 반복횟수, n)으로 변환된다
    # 반복횟수는 위에서 글로벌 변수를 생성할 때 생성자에서 이미 정의했다
    s_prev = repeator(s_prev)

    # a 와 s_prev를 연결한 텐서를 생성하자
    # s_prev : (m, 반복횟수Tx, n_s)
    # a : (m, Tx, 2*n_a)
    # 마지막 축을 기준으로 연결하면 된다
    concat = concatenator([a, s_prev])

    # 중간 에너지 변수 e
    e = densor1(concat)

    # 에너지 변수 energies
    energies = densor2(e)

    # 소프트맥스(에너지) >> attention weights alphas
    alphas = activator(energies)

    # context
    context = dotor([alphas, a])

    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation=softmax)


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    # 원래 존재하는 활성레이어와 돌아온 활성레이어를 연결한다 : merge_mode='concat'
    # 바이디렉셔널 : wrapper클래스 : 기존의 lstm을 변환시킨다
    a = Bidirectional(LSTM(n_a, return_sequences=True), merge_mode='concat')(X)

    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        # 기존의 s, c >> rnn 통과 >> 새로운 s, c
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs, name="model")

    return model


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# 트레이닝
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

model.load_weights('projects/model.h5')

while True:
    day = input("번역을 원하는 날짜를 쓰세요 : ")
    source = string_to_int(day, Tx, human_vocab)
    source = np.array([np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))])
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", day)
    print("output:", ''.join(output))

    attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "September 24 1997", num=7, n_s=64)
    plt.show()

    keep = input("계속 하시겠습니까? y / n : ")
    if keep == "n":
        break
