import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_vectorize(reviews, sentiments):
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
    X = vectorizer.fit_transform(reviews)
    y = np.array(sentiments)
    return X, y, vectorizer
