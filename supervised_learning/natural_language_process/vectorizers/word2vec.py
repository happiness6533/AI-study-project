import numpy as np
from gensim.models import word2vec
import logging


def get_features(words, model, num_features):
    feature_vector = np.zeros((num_features), dtype=np.float32)

    num_words = 0
    index2word_set = set(model.wv.index2word)

    for w in words:
        if w in index2word_set:
            num_words += 1
            feature_vector = np.add(feature_vector, model[w])

    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector


def get_dataset(reviews, model, num_features):
    dataset = list()

    for s in reviews:
        dataset.append(get_features(s, model, num_features))

    reviewFeatureVecs = np.stack(dataset)

    return reviewFeatureVecs


def word2vec_vectorize(reviews, sentiments):
    sentences = []
    for review in reviews:
        sentences.append(review.split())
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = word2vec.Word2Vec(sentences,
                              workers=num_workers, size=num_features,
                              min_count=min_word_count,
                              window=context, sample=downsampling)

    model.save("word2vec_model")
    test_data_vecs = get_dataset(sentences, model, num_features)
    X = test_data_vecs
    y = np.array(sentiments)

    return X, y, model
