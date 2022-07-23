import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from triggerProject.td_utils import *

# 10초 wav 음성파일은
# 1. 44100hz로 분석하면 441000개의 숫자(공기의 압력변화)
# 2. 5511 spectrogram(하나의 spectrogram : 여러 프리퀀시들의 각각의 활성도를 값으로 가지고 있다)
x = graph_spectrogram("audio_examples/example_train.wav")

_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:, 0].shape)
print("Time steps in input after spectrogram", x.shape)

# 10초의 spectrogram 개수
Tx = 5511
# 하나의 spectrogram을 구성하는 프리퀀시들의 개수
n_freq = 101
# output
Ty = 1375

# Load audio segments using pydub : 1초 = 1000ms
activates, negatives, backgrounds = load_raw_audio()


# 파라미터로 주어진 ms길이 만큼의 오디오 데이터 시작위치, 끝위치 선택
def get_random_time_segment(segment_ms):
    # 10초 = 10000ms
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return [segment_start, segment_end]


# 백그라운드에 삽입하려는 소리가 이 전에 삽입했던 소리의 위치와 겹치는지 판단
def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time

    overlap = False
    for previous_start, previous_end in previous_segments:
        if previous_start <= segment_end and previous_end >= segment_start:
            overlap = True
    return overlap


# 백그라운드에 소리 삽입
def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


# 삽입된 음성에 맞추어 y의 라벨을 1로 설정한다
def insert_ones(y, segment_end_ms):
    """
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i <= 1374:
            y[0, i] = 1
    return y


# 트레이닝 샘플 생성
def create_training_example(background, activates, negatives):
    """
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    # Make background quieter
    background = background - 20

    y = np.zeros(shape=[1, Ty])
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    return x, y


x, y = create_training_example(backgrounds[0], activates, negatives)

plt.plot(y[0])

# Load preprocessed training examples
X = np.load("data/XY_train/X.npy")
Y = np.load("data/XY_train/Y.npy")

# Load preprocessed dev set examples
X_dev = np.load("data/XY_dev/X_dev.npy")
Y_dev = np.load("data/XY_dev/Y_dev.npy")


from keras.callbacks import ModelCheckpoint
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers


def model(input_shape):
    X_input = layers.Input(shape=input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = layers.Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = layers.BatchNormalization()(X)
    X = layers.Activation(activation='relu')(X)
    X = layers.Dropout(rate=0.8)(X)

    # Step 2: First GRU Layer (≈4 lines)
    # return_sequences가 True이면 모든 입력에 대해 출력을 생성한다
    # return_sequences가 False이면 마지막에 한번의 출력을 생성한다
    X = layers.GRU(units=128, return_sequences=True)(X)
    X = layers.Dropout(rate=0.8)(X)
    X = layers.BatchNormalization()(X)

    # Step 3: Second GRU Layer (≈4 lines)
    X = layers.GRU(units=128, return_sequences=True)(X)
    X = layers.Dropout(rate=0.8)(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dropout(rate=0.8)(X)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(X)

    model = models.Model(inputs=X_input, outputs=X)

    return model


model = model(input_shape=(Tx, n_freq))
model.summary()

model = models.load_model('./projects/tr_model.h5')
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, Y, batch_size=5, epochs=1)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions


chime_file = "audio_examples/chime.wav"


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')


filename = "data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)

filename = "data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)


# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')


your_filename = "audio_examples/my_audio.wav"

preprocess_audio(your_filename)

chime_threshold = 0.5
prediction = detect_triggerword(your_filename)
chime_on_activate(your_filename, prediction, chime_threshold)
