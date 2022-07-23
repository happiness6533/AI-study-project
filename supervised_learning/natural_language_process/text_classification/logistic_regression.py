import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ..vectorizers.tf_idf import tf_idf_vectorize
from ..vectorizers.word2vec import word2vec_vectorize, get_dataset

DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'
TRAIN_CLEAN_DATA = 'train_clean.csv'
RANDOM_SEED = 42
TEST_SPLIT = 0.2

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

# tf-idf를 활용한 문장 벡터화
X, y, vectorizer = tf_idf_vectorize(reviews, sentiments)

features = vectorizer.get_feature_names()
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)
predicted = lgs.predict(X_eval)
print("Accuracy: %f" % lgs.score(X_eval, y_eval))

# 캐글 제출
TEST_CLEAN_DATA = 'test_clean.csv'
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
testDataVecs = vectorizer.transform(test_data['review'])
test_predicted = lgs.predict(testDataVecs)
print(test_predicted)
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
answer_dataset = pd.DataFrame({'id': test_data['id'], 'sentiment': test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_tfidf_answer.csv', index=False, quoting=3)

# 2. word2vec을 활용한 문장 벡터화
X, y, model = word2vec_vectorize(reviews, sentiments)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)
print("Accuracy: %f" % lgs.score(X_eval, y_eval))

# 캐글 제출
TEST_CLEAN_DATA = 'test_clean.csv'
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
test_review = list(test_data['review'])
test_data.head(5)
test_sentences = list()
for review in test_review:
    test_sentences.append(review.split())
test_data_vecs = get_dataset(test_sentences, model, num_features=300)
DATA_OUT_PATH = './data_out/'

test_predicted = lgs.predict(test_data_vecs)
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
ids = list(test_data['id'])
answer_dataset = pd.DataFrame({'id': ids, 'sentiment': test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_w2v_answer.csv', index=False, quoting=3)
