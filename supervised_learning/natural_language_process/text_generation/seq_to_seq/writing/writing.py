from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import sys
import io

from writingProject.utils import *
from writingProject.shakespeare_utils import *
import random


# 그래디언트 클립
def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


# 샘플링
def sample(parameters, char_to_ix):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # x, a_prev 초기화
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []

    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']
    while idx != newline_character and counter != 50:
        a = np.tanh(np.matmul(Waa, a_prev) + np.matmul(Wax, x) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)

        # np.arange(0, n) : 0 부터 n - 1까지 정렬
        # np.random.choice(넘파이배열, 확률분포) : 샘플링
        idx = np.random.choice(np.arange(0, vocab_size), p=y.ravel())

        # 샘플링된 인덱스 추가
        indices.append(idx)

        # one hot encoding
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # a_prev를 다음 레이어로 전송
        a_prev = a

        # 철자의 길이 제한
        counter += 1

    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices


# single optimization
def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size=27)

    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)

    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


# rnn model
def model(data, ix_to_char, char_to_ix, num_iterations=35000, n_a=50, dino_names=7, vocab_size=27):
    """
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    parameters -- learned parameters
    """
    n_x, n_y = vocab_size, vocab_size

    parameters = initialize_parameters(n_a, n_x, n_y)

    # cost를 convex하게 세팅한다
    loss = get_initial_loss(vocab_size, dino_names)

    # 파일 읽어서 가져오기
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle
    np.random.shuffle(examples)

    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):
        # index <= #트레이닝셋
        # x0 = 0, x1이후로 샘플링
        # y = x1이후와 동일, 마지막은 "\n"
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one. 

            print('\n')

    return parameters


# 데이터
data = open('dinos.txt', 'r').read()

# 소문자로 변환
data = data.lower()

# set으로 변환(27개 = 알파벳 26개 + "\n")
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

# 사전 생성
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
print(ix_to_char)

# 트레이닝
parameters = model(data, ix_to_char, char_to_ix)


# 셰익스피어
from writingProject.shakespeare_utils import *
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
generate_output()
