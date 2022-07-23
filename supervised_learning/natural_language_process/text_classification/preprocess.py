import os
import re
import json
import html5lib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import konlpy


def preprocessing(review, remove_stopwords=False):
    # 불용어 제거는 옵션으로 선택 가능하다.

    # 1. HTML 태그 제거
    review_text = BeautifulSoup(review, "html5lib").get_text()

    # 2. 영어가 아닌 특수문자들을 공백(" ")으로 바꾸기
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. 대문자들을 소문자로 바꾸고 공백단위로 텍스트들 나눠서 리스트로 만든다.
    words = review_text.lower().split()

    if remove_stopwords:
        # 4. 불용어들을 제거

        # 영어에 관련된 불용어 불러오기
        stops = set(stopwords.words("english"))
        # 불용어가 아닌 단어들로 이루어진 새로운 리스트 생성
        words = [w for w in words if not w in stops]
        # 5. 단어 리스트를 공백을 넣어서 하나의 글로 합친다.
        clean_review = ' '.join(words)

    else:  # 불용어 제거하지 않을 때
        clean_review = ' '.join(words)

    return clean_review


def preprocessing_korean(review, remove_stopwords=False):
    # 불용어 제거는 옵션으로 선택 가능하다.
    if type(review) != str:
        return []
    # 2. 영어가 아닌 특수문자들을 공백(" ")으로 바꾸기
    review_text = re.sub("[^가-힇 ㄱ-ㅎ ㅏ-ㅣ \\s]", " ", review)
    okt = Okt()
    word_review = okt.morphs(review_text, stem=True)
    stopwords = ['은', "는", "이", "가", "하", "아", "것", "들", "의", "있", "되", "수", "보", "주", "등", "한"]
    if remove_stopwords:
        # 4. 불용어
        word_review = [token for token in word_review if not token in stopwords]

    return word_review


# 훈련 데이터
def save_files(name_of_train_data, name_of_test_data):
    DATA_IN_PATH = 'data_in/'
    train_data = pd.read_csv(DATA_IN_PATH + name_of_train_data, header=0, delimiter='\t', quoting=3)
    print(train_data['review'][0])
    clean_train_reviews = [preprocessing(review, remove_stopwords=True) for review in train_data['review']]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_train_reviews)
    text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
    print(text_sequences[0])
    word_vocab = tokenizer.word_index
    word_vocab["<PAD>"] = 0
    print("전체 단어 개수: ", len(word_vocab))

    train_inputs = pad_sequences(text_sequences, maxlen=174, padding='post')
    print('Shape of train data: ', train_inputs.shape)
    train_labels = np.array(train_data['sentiment'])
    print('Shape of label tensor:', train_labels.shape)

    np.save(open(DATA_IN_PATH + 'train_input.npy', 'wb'), train_inputs)  # 전처리 된 데이터를 넘파이 형태로 저장 => 단어 인덱스로 문장 구성
    np.save(open(DATA_IN_PATH + 'train_label.npy', 'wb'), train_labels)  # 전처리 된 데이터를 넘파이 형태로 저장
    clean_train_df = pd.DataFrame({'review': clean_train_reviews, 'sentiment': train_data['sentiment']})
    clean_train_df.to_csv(DATA_IN_PATH + 'train_clean.csv', index=False)  # 정제된 텍스트를 csv 형태로 저장 => 텍스트로 문장 구성

    data_configs = {}
    data_configs['vocab'] = word_vocab
    data_configs['vocab_size'] = len(word_vocab)
    json.dump(data_configs, open(DATA_IN_PATH + 'data_configs.json', 'w'), ensure_ascii=False)  # 데이터 사전을 json 형태로 저장

    # 테스트 데이터
    test_data = pd.read_csv(DATA_IN_PATH + name_of_test_data, header=0, delimiter="\t", quoting=3)
    clean_test_reviews = [preprocessing(review, remove_stopwords=True) for review in test_data['review']]
    clean_test_df = pd.DataFrame({'review': clean_test_reviews, 'id': test_data['id']})
    test_id = np.array(test_data['id'])
    text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)
    test_inputs = pad_sequences(text_sequences, maxlen=174, padding='post')
    TEST_INPUT_DATA = 'test_input.npy'
    TEST_CLEAN_DATA = 'test_clean.csv'
    TEST_ID_DATA = 'test_id.npy'
    np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
    np.save(open(DATA_IN_PATH + TEST_ID_DATA, 'wb'), test_id)
    clean_test_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index=False)


def save_korean_files(name_of_train_data, name_of_test_data):
    DATA_IN_PATH = './data_in/'
    train_data = pd.read_csv(DATA_IN_PATH + name_of_train_data, header=0, delimiter='\t', quoting=3)
    clean_train_reviews = [preprocessing_korean(review, remove_stopwords=True) for review in train_data['document']]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(clean_train_reviews)
    text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
    print(text_sequences[0])
    word_vocab = tokenizer.word_index
    word_vocab["<PAD>"] = 0
    print("전체 단어 개수: ", len(word_vocab))

    train_inputs = pad_sequences(text_sequences, maxlen=8, padding='post')
    print('Shape of train data: ', train_inputs.shape)
    train_labels = np.array(train_data['sentiment'])
    print('Shape of label tensor:', train_labels.shape)

    np.save(open(DATA_IN_PATH + 'train_input.npy', 'wb'), train_inputs)  # 전처리 된 데이터를 넘파이 형태로 저장 => 단어 인덱스로 문장 구성
    np.save(open(DATA_IN_PATH + 'train_label.npy', 'wb'), train_labels)  # 전처리 된 데이터를 넘파이 형태로 저장
    pd.DataFrame({'review': clean_train_reviews, 'sentiment': train_data['sentiment']}).to_csv(
        DATA_IN_PATH + 'train_clean.csv', index=False
    )  # 정제된 텍스트를 csv 형태로 저장 => 텍스트로 문장 구성

    data_configs = {}
    data_configs['vocab'] = word_vocab
    data_configs['vocab_size'] = len(word_vocab)
    json.dump(data_configs, open(DATA_IN_PATH + 'data_configs.json', 'w'), ensure_ascii=False)  # 데이터 사전을 json 형태로 저장

    # 테스트 데이터
    test_data = pd.read_csv(DATA_IN_PATH + name_of_test_data, header=0, delimiter="\t", quoting=3)
    clean_test_reviews = [preprocessing(review, remove_stopwords=True) for review in test_data['document']]
    clean_test_df = pd.DataFrame({'review': clean_test_reviews, 'id': test_data['id']})
    test_id = np.array(test_data['id'])
    text_sequences = tokenizer.texts_to_sequences(clean_test_reviews)
    test_inputs = pad_sequences(text_sequences, maxlen=8, padding='post')
    TEST_INPUT_DATA = 'test_input.npy'
    TEST_CLEAN_DATA = 'test_clean.csv'
    TEST_ID_DATA = 'test_id.npy'
    np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
    np.save(open(DATA_IN_PATH + TEST_ID_DATA, 'wb'), test_id)
    clean_test_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index=False)


if __name__ == "__main__":
    save_files("labeledTrainData.tsv", "testData.tsv")
    save_files("labeledTrainData.tsv", "testData.tsv")
    # save_korean_files("ratings.txt", "testData.tsv")
    # save_korean_files("labeledTrainDataKorean.tsv", "testData.tsv")
