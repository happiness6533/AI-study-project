import numpy as np
from emojifyProject.emo_utils import *
import emoji
import matplotlib.pyplot as plt

# 데이터
# train : 127개 / test : 56개
# x : 문장 / y : 이모지
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=len).split())

# 데이터 예시 확인
index = 1
print(X_train[index], label_to_emoji(Y_train[index]))

# 원 핫 인코딩
# y는 원래 (m, 1)이며, (m, 5)로 원핫인코딩한다
Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)

index = 50
print(Y_train[index], "is converted into one hot", Y_oh_train[index])

# 워드디멘션
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# You've loaded:
# 워드 >> 인덱스 :사전 : 단어 >> 인덱스로 매핑
# 400,001 words, with the valid indices ranging from 0 to 400,000
# 파라미터 개수 : 40만1 * 50차원 = 2천50만개 : 트레이닝 안하고 바로 쓴다
# index_to_word: 인덱스 >> 사전의 단어로 매핑하는 딕셔너리
# word_to_vec_map: 단어 >> 임베딩벡터로 매핑하는 딕셔너리
word = "cucumber"
index = 289846
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(index) + "th word in the vocabulary is", index_to_word[index])


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(shape=word_to_vec_map[words[0]].shape)

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)

    return avg


avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = ", avg)


# 단어벡터들의 평균들 >> 선형변환
# 선형변환 >> 소프트맥스
# 소프트맥스 >> loss함수
# loss최적화
def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    # Define number of training examples
    m = Y.shape[0]  # number of training examples
    n_y = 5  # number of classes
    n_h = 50  # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.matmul(W, avg) + b
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A"
            # 소프트맥스 코스트 함수!
            cost = -np.sum(Y_oh * np.log(a), axis=0, keepdims=True)

            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


print(X_train.shape)
print(Y_train.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
Y = np.asarray([5, 0, 0, 5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
print(Y.shape)

X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
                'Lets go party and drinks', 'Congrats on the new job', 'Congratulations',
                'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
                'You totally deserve this prize', 'Let us go play football',
                'Are you down for football this afternoon', 'Work hard play harder',
                'It is suprising how people can be dumb sometimes',
                'I am very disappointed', 'It is the best day in my life',
                'I think I will end up alone', 'My life is so boring', 'Good job',
                'Great so awesome'])

print(X.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(type(X_train))

pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)

# ### 1.4 - Examining test set performance
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

# adore라는 단어는 트레이닝셋에 없지만
# 단어공간의 추론성에 의해, 하트 이모지로 연결될 수 있다
X_my_sentences = np.array(
    ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)

# confusion matrix 생성해보기
print(Y_test.shape)
print('           ' + label_to_emoji(0) + '    ' + label_to_emoji(1) + '    ' + label_to_emoji(
    2) + '    ' + label_to_emoji(3) + '   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56, ), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)







# lstm을 통과해서 이모지파이어 만들어 보기
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform


# 우선, lstm에 문장을 넣어야 한다
# 문장의 길이가 모두 다르기 때문에, 가지고 있는 트레이닝셋의 최고 길이 문장을 기준으로
# zero-padding 해서, 모든 문장의 길이를 동일하게 맞춘다

# 케라스에서는 임베딩 매트릭스를 하나의 레이어로 간주한다
# 케라스로 임베딩 레이어를 생성하고, 가지고 있는 pretrained된 벡터공간을 이니셜라이즈로 할 수 있다
# 이 레이어를 fix 할수도 있고, update 할수도 있다

# 임베딩 레이어의 인풋 : (배치사이즈 m, 최대길이로 패딩된 정수배열의 길이)
# 임베딩 레이어의 아웃풋 : (배치사이즈 m, 임베딩 벡터 개수, 50차원)
def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j += 1

    return X_indices


X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
print("X1 =", X1)
print("X1_indices =", X1_indices)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    # 다운받은 임베딩 벡터들을 케라스 레이어에 옮겨보자!
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    # Initialize the embedding matrix
    # as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros(shape=(vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix
    # to be the word vector representation of the "index"th word of the vocabulary
    # 프리트레인된 다운로드한 워드임베딩 결과를
    # 방금 생성한 emb_matrix에 집어넣는다
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer
    # with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False : 임베딩 상태가 fix된다
    # 만약 trainable = True 로 설정하면, 트레이닝 결과에 의해 임베딩 상태가 변화한다
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)

    # Build the embedding layer
    # it is required before setting the weights of the embedding layer.
    # Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix
    # Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])



def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    # 문장 >> 인덱스 변환 >> 임베딩 벡터집합 변환 >> 임베딩집합을 lstm에 인풋
    # lstm에서 나온 결과를 드랍아웃
    # 결과를 다시 lstm에 인풋 >> 결과 1개로 압축
    # 1개의 결과 >> 드랍아웃 >> 덴스레이어 >> 소프트맥스 : 결과(이모티콘 벡터)

    # 텐서 생성
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    # 액티베이션 레이어 128
    # 리턴에서 배치 시퀀스 유지, 즉 모든 셀에서 값을 출력
    X = LSTM(units=128, return_sequences=True)(embeddings)

    X = Dropout(rate=0.5)(X)

    # 액티베이션 레이어 128
    # 리턴에서 배치 시퀀스 유지안함, 즉 마지막 셀에서 하나의 값만 출력
    X = LSTM(units=128, return_sequences=False)(X)

    X = Dropout(rate=0.5)(X)

    # 아웃풋 차원 5
    X = Dense(units=5)(X)

    # 소프트맥스
    X = Activation(activation='softmax')(X)

    # 모델 생성
    model = Model(inputs=sentence_indices, outputs=X, name="model")

    return model


model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 데이터
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=5)

# 트레이닝
model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

# 테스트셋 정확도
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C=5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

# 테스트셋 해보기
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if (num != Y_test[i]):
        print('Expected emoji:' + label_to_emoji(Y_test[i]) + ' prediction: ' + X_test[i] + label_to_emoji(num).strip())

# my test
while True:
    mySentence = input("이모티콘을 붙이고 싶은 문장을 쓰세요 : ")

    x_test = np.array([mySentence])
    X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    print(x_test[0] + ' ' + label_to_emoji(np.argmax(model.predict(X_test_indices))))
    more = input("더 하시겠습니까? y / n : ")
    if more == "n":
        break
