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


from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)

k_means.fit(train_input)
k_means.labels_
print("0 cluster:", train_label[k_means.labels_ == 0])
print("1 cluster:", train_label[k_means.labels_ == 1])
print("2 cluster:", train_label[k_means.labels_ == 2])

import numpy as np
new_input  = np.array([[6.1, 2.8, 4.7, 1.2]])

prediction = k_means.predict(new_input)
print(prediction)


predict_cluster = k_means.predict(test_input)
print(predict_cluster)


np_arr = np.array(predict_cluster)
np_arr[np_arr==0], np_arr[np_arr==1], np_arr[np_arr==2] = 3, 4, 5
np_arr[np_arr==3] = 1
np_arr[np_arr==4] = 0
np_arr[np_arr==5] = 2
predict_label = np_arr.tolist()
print(predict_label)

print('test accuracy {:.2f}'.format(np.mean(predict_label == test_label)))
