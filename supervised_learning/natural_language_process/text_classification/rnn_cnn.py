from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
# 텍스트를 인자로 넘겨받아서 여기서 처리해야 되는구나
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
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
clean_train_reviews = [preprocessing(review, remove_stopwords=True) for review in ["hello friends"]]
print(clean_train_reviews)
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
print(text_sequences)
