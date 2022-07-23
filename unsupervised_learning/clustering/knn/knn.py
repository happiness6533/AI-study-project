import sklearn
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("iris_dataset key: {}".format(iris_dataset.keys()))
print(iris_dataset['data'])
print("shape of data: {}".format(iris_dataset['data'].shape))
print(iris_dataset['feature_names'])

print(iris_dataset['target'])
print(iris_dataset['target_names'])
print(iris_dataset['DESCR'])

target = iris_dataset['target']
from sklearn.model_selection import train_test_split

train_input, test_input, train_label, test_label = train_test_split(iris_dataset['data'],
                                                                    target,
                                                                    test_size=0.25,
                                                                    random_state=42)

print("shape of train_input: {}".format(train_input.shape))
print("shape of test_input: {}".format(test_input.shape))
print("shape of train_label: {}".format(train_label.shape))
print("shape of test_label: {}".format(test_label.shape))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_input, train_label)
import numpy as np

new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
knn.predict(new_input)

predict_label = knn.predict(test_input)
print(predict_label)
print('test accuracy {:.2f}'.format(np.mean(predict_label == test_label)))
