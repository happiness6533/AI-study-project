import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

if __name__ == "__main__":
    # 1. 데이터 준비
    DATA_IN_PATH = 'data_in/'
    file_list = ['labeledTrainData.tsv.zip', 'unlabeledTrainData.tsv.zip', 'testData.tsv.zip']

    for file in file_list:
        zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
        zipRef.extractall(DATA_IN_PATH)
        zipRef.close()

    train_data = pd.read_csv(DATA_IN_PATH + 'labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    print(train_data.head())

    # 2. 데이터 개수: 훈련에 충분한 양인지 확인하고 부족하다면 보강
    print('전체 학습데이터의 개수: {}'.format(len(train_data)))

    # 1. 데이터 상태 분석
    cloud = WordCloud(background_color='black', width=800, height=600)\
        .generate(" ".join(train_data['review']))
    plt.figure(figsize=(15, 10))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()

    # 2. 데이터 균형 분석
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(train_data['sentiment'])
    print("긍정 리뷰 개수: {}".format(train_data['sentiment'].value_counts()[1]))
    print("부정 리뷰 개수: {}".format(train_data['sentiment'].value_counts()[0]))

    # 3. 데이터 특징 분석: 이상치 확인 및 평균을 기준으로 최대값 결정
    train_length = train_data['review'].apply(len)
    train_length.head()
    plt.figure(figsize=(12, 5))
    plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
    plt.yscale('log', nonposy='clip')
    plt.title('Log-Histogram of length of review')
    plt.xlabel('Length of review')
    plt.ylabel('Number of review')
    plt.show()
    print('리뷰 길이 최대 값: {}'.format(np.max(train_length)))
    print('리뷰 길이 최소 값: {}'.format(np.min(train_length)))
    print('리뷰 길이 평균 값: {:.2f}'.format(np.mean(train_length)))
    print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
    print('리뷰 길이 중간 값: {}'.format(np.median(train_length)))

    plt.figure(figsize=(12, 5))
    plt.boxplot(train_length, labels=['counts'], showmeans=True)
    plt.show()
    print('리뷰 길이 제 1 사분위: {}'.format(np.percentile(train_length, 25)))
    print('리뷰 길이 제 3 사분위: {}'.format(np.percentile(train_length, 75)))

    train_word_counts = train_data['review'].apply(lambda x: len(x.split(' ')))
    plt.figure(figsize=(15, 10))
    plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
    plt.title('Log-Histogram of word count in review', fontsize=15)
    plt.yscale('log', nonposy='clip')
    plt.legend()
    plt.xlabel('Number of words', fontsize=15)
    plt.ylabel('Number of reviews', fontsize=15)
    plt.show()
    print('리뷰 단어 개수 최대 값: {}'.format(np.max(train_word_counts)))
    print('리뷰 단어 개수 최소 값: {}'.format(np.min(train_word_counts)))
    print('리뷰 단어 개수 평균 값: {:.2f}'.format(np.mean(train_word_counts)))
    print('리뷰 단어 개수 표준편차: {:.2f}'.format(np.std(train_word_counts)))
    print('리뷰 단어 개수 중간 값: {}'.format(np.median(train_word_counts)))

    plt.figure(figsize=(12, 5))
    plt.boxplot(train_word_counts, labels=['word counts'], showmeans=True)
    plt.show()
    print('리뷰 단어 개수 제 1 사분위: {}'.format(np.percentile(train_word_counts, 25)))
    print('리뷰 단어 개수 제 3 사분위: {}'.format(np.percentile(train_word_counts, 75)))

    # 특수문자 및 대, 소문자 비율
    qmarks = np.mean(train_data['review'].apply(lambda x: '?' in x))  # 물음표가 구두점으로 쓰임
    fullstop = np.mean(train_data['review'].apply(lambda x: '.' in x))  # 마침표
    capital_first = np.mean(train_data['review'].apply(lambda x: x[0].isupper()))  # 첫번째 대문자
    capitals = np.mean(train_data['review'].apply(lambda x: max([y.isupper() for y in x])))  # 대문자가 몇개
    numbers = np.mean(train_data['review'].apply(lambda x: max([y.isdigit() for y in x])))  # 숫자가 몇개
    print('물음표가있는 질문: {:.2f}%'.format(qmarks * 100))
    print('마침표가 있는 질문: {:.2f}%'.format(fullstop * 100))
    print('첫 글자가 대문자 인 질문: {:.2f}%'.format(capital_first * 100))
    print('대문자가있는 질문: {:.2f}%'.format(capitals * 100))
    print('숫자가있는 질문: {:.2f}%'.format(numbers * 100))
