import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pymysql


class RecommendAgent:
    def __init__(self, num_of_movies, size_of_movie_vector, dists_of_movie_similarity, size_of_pos_training_sample,
                 dist_of_movie_frequency, size_of_neg_training_sample, learning_rate, dists_of_movie_choice_by_user,
                 num_of_movie_vectors_to_make_input_vector_of_user_network, num_of_users, score_matrix, const_lambda,
                 alpha, beta, num_of_training, path_of_save, database_user, database_password, database_name):
        self.num_of_movies = num_of_movies
        self.size_of_movie_vector = size_of_movie_vector
        self.embedded_movie_vectors = self.build_one_hot_movie_vector_to_embedded_movie_vector_network()

        self.dists_of_movie_similarity = dists_of_movie_similarity
        self.size_of_pos_training_sample = size_of_pos_training_sample
        self.dist_of_movie_frequency = dist_of_movie_frequency
        self.size_of_neg_training_sample = size_of_neg_training_sample
        self.learning_rate = learning_rate
        self.cost_of_negative_sampling_network = self.build_negative_sampling_network_1()

        self.dists_of_movie_choice_by_user = dists_of_movie_choice_by_user
        self.num_of_movie_vectors_to_make_input_vector_of_user_network = num_of_movie_vectors_to_make_input_vector_of_user_network
        self.num_of_users = num_of_users
        self.input_vectors_of_user_network = self.make_input_vectors_of_user_network()

        self.size_of_user_vector = self.size_of_movie_vector
        self.size_of_input_vector_of_user_network = self.num_of_movie_vectors_to_make_input_vector_of_user_network * self.size_of_movie_vector
        self.embedded_user_vectors = self.build_user_network()

        self.score_matrix = score_matrix
        self.const_lambda = const_lambda
        self.predicted_score_matrix, self.cost_of_collaborative_filtering_network = self.build_collaborative_filtering_network()

        self.alpha = alpha
        self.beta = beta
        self.cost_of_two_networks, self.optimizer_of_two_networks = self.make_cost_and_optimizer_of_two_networks()

        self.num_of_training = num_of_training

        self.path_of_save = path_of_save

        self.database_user = database_user
        self.database_password = database_password
        self.database_name = database_name

    def build_one_hot_movie_vector_to_embedded_movie_vector_network(self):
        one_hot_movie_vectors = tf.constant(np.eye(self.num_of_movies))
        embedded_movie_vector_parameters_variable = tf.Variable(
            np.random.rand(self.num_of_movies, self.size_of_movie_vector),
            name="embedded_movie_vector_parameters_variable")
        embedded_movie_vectors = tf.matmul(one_hot_movie_vectors, embedded_movie_vector_parameters_variable)

        return embedded_movie_vectors

    def build_negative_sampling_network_1(self):
        pos_training_samples = []
        neg_training_samples = []
        for i in range(self.num_of_movies):
            # pos_training_sample
            dist_of_similarity_by_movie = self.dists_of_movie_similarity[i]
            dist_of_similarity_by_movie = np.array(dist_of_similarity_by_movie)
            dist_of_similarity_by_movie = dist_of_similarity_by_movie / dist_of_similarity_by_movie.sum()
            indexes_of_pos_training_sample = np.random.choice(self.num_of_movies,
                                                              size=self.size_of_pos_training_sample,
                                                              p=dist_of_similarity_by_movie)
            start_index = indexes_of_pos_training_sample[0]
            pos_training_sample = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                           [1, self.size_of_movie_vector])
            for j in range(1, self.size_of_pos_training_sample):
                index = indexes_of_pos_training_sample[j]
                pos_training_sample = tf.concat([pos_training_sample,
                                                 tf.slice(self.embedded_movie_vectors, [index, 0],
                                                          [1, self.size_of_movie_vector])], axis=0)
            pos_training_sample = tf.stop_gradient(tf.transpose(pos_training_sample))  # 스탑 그레디언트?
            pos_training_samples.append(pos_training_sample)

            # neg_training_sample
            dist_of_movie_frequency = np.array(self.dist_of_movie_frequency)
            dist_of_movie_frequency = dist_of_movie_frequency / dist_of_movie_frequency.sum()
            indexes_of_neg_training_samples = np.random.choice(self.num_of_movies,
                                                               size=self.size_of_neg_training_sample,
                                                               p=dist_of_movie_frequency)
            start_index = indexes_of_neg_training_samples[0]
            neg_training_sample = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                           [1, self.size_of_movie_vector])
            for j in range(1, self.size_of_neg_training_sample):
                index = indexes_of_neg_training_samples[j]
                neg_training_sample = tf.concat([neg_training_sample,
                                                 tf.slice(self.embedded_movie_vectors, [index, 0],
                                                          [1, self.size_of_movie_vector])], axis=0)
            neg_training_sample = tf.stop_gradient(tf.transpose(neg_training_sample))  # 스탑 그레디언트?
            neg_training_samples.append(neg_training_sample)

        # cost1 : 로그0 주의!
        cost_of_negative_sampling_network = tf.to_double(tf.constant(0))
        for pos_training_sample in pos_training_samples:
            cost_of_negative_sampling_network = cost_of_negative_sampling_network + tf.reduce_mean(tf.log(
                tf.clip_by_value(tf.sigmoid(tf.matmul(self.embedded_movie_vectors, pos_training_sample)), 0.00001,
                                 0.99999)))

        for neg_training_sample in neg_training_samples:
            cost_of_negative_sampling_network = cost_of_negative_sampling_network + tf.reduce_mean(
                tf.log(tf.clip_by_value(1 - tf.sigmoid(tf.matmul(self.embedded_movie_vectors, neg_training_sample)),
                                        0.00001, 0.99999)))
        size_of_sample = self.size_of_pos_training_sample + self.size_of_neg_training_sample
        cost_of_negative_sampling_network = -(cost_of_negative_sampling_network / size_of_sample)  # 부호 체크!

        return cost_of_negative_sampling_network

    def make_input_vectors_of_user_network(self):
        dist_of_movie_choice_by_user = self.dists_of_movie_choice_by_user[0]
        dist_of_movie_choice_by_user = np.array(dist_of_movie_choice_by_user)
        dist_of_movie_choice_by_user = dist_of_movie_choice_by_user / dist_of_movie_choice_by_user.sum()
        indexes_of_movie_vectors = np.random.choice(self.num_of_movies,
                                                    size=self.num_of_movie_vectors_to_make_input_vector_of_user_network,
                                                    p=dist_of_movie_choice_by_user)
        start_index = indexes_of_movie_vectors[0]
        input_vector_of_user_network = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                                [1, self.size_of_movie_vector])
        for i in range(1, self.num_of_movie_vectors_to_make_input_vector_of_user_network):
            index = indexes_of_movie_vectors[i]
            input_vector_of_user_network = tf.concat([input_vector_of_user_network,
                                                      tf.slice(self.embedded_movie_vectors, [index, 0],
                                                               [1, self.size_of_movie_vector])], axis=1)

        input_vectors_of_user_network = input_vector_of_user_network
        for i in range(1, self.num_of_users):
            dist_of_movie_choice_by_user = self.dists_of_movie_choice_by_user[i]
            dist_of_movie_choice_by_user = np.array(dist_of_movie_choice_by_user)
            dist_of_movie_choice_by_user = dist_of_movie_choice_by_user / dist_of_movie_choice_by_user.sum()
            indexes_of_movie_vectors = np.random.choice(self.num_of_movies,
                                                        size=self.num_of_movie_vectors_to_make_input_vector_of_user_network,
                                                        p=dist_of_movie_choice_by_user)
            start_index = indexes_of_movie_vectors[0]
            input_vector_of_user_network = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                                    [1, self.size_of_movie_vector])

            for j in range(1, self.num_of_movie_vectors_to_make_input_vector_of_user_network):
                index = indexes_of_movie_vectors[j]
                input_vector_of_user_network = tf.concat([input_vector_of_user_network,
                                                          tf.slice(self.embedded_movie_vectors, [index, 0],
                                                                   [1, self.size_of_movie_vector])], axis=1)
            input_vectors_of_user_network = tf.concat([input_vectors_of_user_network, input_vector_of_user_network],
                                                      axis=0)
        return input_vectors_of_user_network

    def build_user_network(self):
        self.input_vectors_of_user_network_placeholder = tf.placeholder(dtype=tf.double, shape=[None,
                                                                                                self.size_of_input_vector_of_user_network],
                                                                        name="input_vectors_of_user_network_placeholder")

        layer1 = tf.layers.dense(inputs=self.input_vectors_of_user_network_placeholder, units=self.size_of_user_vector,
                                 activation=None)
        layer2 = tf.nn.leaky_relu(layer1)
        layer3 = tf.nn.dropout(layer2, keep_prob=0.75)

        layer4 = tf.layers.dense(inputs=layer3, units=self.size_of_user_vector, activation=None)
        layer5 = tf.nn.leaky_relu(layer4)
        layer6 = tf.nn.dropout(layer5, keep_prob=0.75)

        layer7 = tf.layers.dense(inputs=layer6, units=self.size_of_user_vector, activation=None)
        layer8 = tf.nn.leaky_relu(layer7)
        layer9 = tf.nn.dropout(layer8, keep_prob=0.75)

        layer10 = tf.layers.dense(inputs=layer9, units=self.size_of_user_vector, activation=None)
        layer11 = tf.nn.leaky_relu(layer10)
        layer12 = tf.nn.dropout(layer11, keep_prob=0.75)

        layer13 = tf.layers.dense(inputs=layer12, units=self.size_of_user_vector, activation=None)

        embedded_user_vectors = layer13

        return embedded_user_vectors

    def build_collaborative_filtering_network(self):
        score_matrix = np.array(self.score_matrix)
        score_matrix = tf.to_double(score_matrix)
        score_matrix = tf.reshape(score_matrix, shape=[-1])

        predicted_score_matrix = tf.matmul(self.embedded_user_vectors, tf.transpose(self.embedded_movie_vectors))
        predicted_score_matrix = tf.reshape(predicted_score_matrix, shape=[-1])
        predicted_score_matrix = predicted_score_matrix

        cost_of_collaborative_filtering_network = tf.reduce_mean(tf.square(score_matrix - predicted_score_matrix))

        # cost2 : regularize 주의!
        cost_of_collaborative_filtering_network = 0.5 * (cost_of_collaborative_filtering_network + self.const_lambda * (
                tf.reduce_mean(self.embedded_user_vectors) + tf.reduce_mean(self.embedded_movie_vectors)))

        return predicted_score_matrix, cost_of_collaborative_filtering_network

    def make_cost_and_optimizer_of_two_networks(self):
        cost_of_two_networks = self.alpha * self.cost_of_negative_sampling_network + self.beta * self.cost_of_collaborative_filtering_network
        trainer_of_two_networks = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        tuple_of_gradient_and_variable = trainer_of_two_networks.compute_gradients(cost_of_two_networks)

        # gradient explode 주의!
        list_of_clipped_gradient_and_variable = []
        for gradient, variable in tuple_of_gradient_and_variable:
            list_of_clipped_gradient_and_variable.append(
                (tf.clip_by_value(gradient, -10, 10), variable))  # 그레디언트 제한의 기준?
        tuple_of_clipped_gradient_and_variable = tuple(list_of_clipped_gradient_and_variable)

        optimizer_of_two_networks = trainer_of_two_networks.apply_gradients(tuple_of_clipped_gradient_and_variable)

        return cost_of_two_networks, optimizer_of_two_networks

    def learn(self):
        saver = tf.train.Saver()

        sess = tf.Session()
        reset_learn = input("처음부터 다시 학습하시겠습니까?(y/n) : ")
        if reset_learn == "y":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, self.path_of_save)
            print("Model restored from %s" % (self.path_of_save))

        costs_of_two_network = []
        costs_of_negative_sampling_network = []
        costs_of_collaborative_filtering_network = []
        for i in range(self.num_of_training):
            input_vectors_of_user_network = sess.run(self.input_vectors_of_user_network)
            cost_of_two_network, _ = sess.run([self.cost_of_two_networks, self.optimizer_of_two_networks], feed_dict={
                self.input_vectors_of_user_network_placeholder: input_vectors_of_user_network})

            if i % 50 == 0:
                cost_of_negative_sampling_network = sess.run(self.cost_of_negative_sampling_network, feed_dict={
                    self.input_vectors_of_user_network_placeholder: input_vectors_of_user_network})
                cost_of_collaborative_filtering_network = sess.run(self.cost_of_collaborative_filtering_network,
                                                                   feed_dict={
                                                                       self.input_vectors_of_user_network_placeholder: input_vectors_of_user_network})
                costs_of_two_network.append(cost_of_two_network)
                costs_of_negative_sampling_network.append(cost_of_negative_sampling_network)
                costs_of_collaborative_filtering_network.append(cost_of_collaborative_filtering_network)

                print("%d번째 학습이 진행중입니다." % (i))
                print("현재의 cost_of_two_network : %f" % (cost_of_two_network))
                print("현재의 costs_of_negative_sampling_network : %f" % (cost_of_negative_sampling_network))
                print("현재의 costs_of_collaborative_filtering_network : %f" % (cost_of_collaborative_filtering_network))
                print(sess.run(self.predicted_score_matrix, feed_dict={
                    self.input_vectors_of_user_network_placeholder: input_vectors_of_user_network}))

                # save
                saver.save(sess, self.path_of_save)
                print("Model saved in %s" % (self.path_of_save))

        sess.close()

        # plot the cost_of_two_networks
        plt.plot(np.squeeze(costs_of_two_network))
        plt.ylabel("cost_of_two_network")
        plt.title("learn")
        plt.show()

        # plot the cost_of_negative_sampling_network
        plt.plot(np.squeeze(costs_of_negative_sampling_network))
        plt.ylabel("cost_of_negative_sampling_network")
        plt.title("learn")
        plt.show()

        # plot the costs_of_collaborative_filtering_network
        plt.plot(np.squeeze(costs_of_collaborative_filtering_network))
        plt.ylabel("costs_of_collaborative_filtering_network")
        plt.title("learn")
        plt.show()

    def build_negative_sampling_network_2(self):

    def build_negative_sampling_network_2(self):
        pos_training_samples = []
        neg_training_samples = []
        for i in range(self.num_of_movies):
            # pos_training_sample
            dist_of_similarity_by_movie = self.dists_of_movie_similarity[i]
            dist_of_similarity_by_movie = np.array(dist_of_similarity_by_movie)
            dist_of_similarity_by_movie = dist_of_similarity_by_movie / dist_of_similarity_by_movie.sum()
            indexes_of_pos_training_sample = np.random.choice(self.num_of_movies,
                                                              size=self.size_of_pos_training_sample,
                                                              p=dist_of_similarity_by_movie)
            start_index = indexes_of_pos_training_sample[0]
            pos_training_sample = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                           [1, self.size_of_movie_vector])
            for j in range(1, self.size_of_pos_training_sample):
                index = indexes_of_pos_training_sample[j]
                pos_training_sample = tf.concat([pos_training_sample,
                                                 tf.slice(self.embedded_movie_vectors, [index, 0],
                                                          [1, self.size_of_movie_vector])], axis=0)
            pos_training_sample = tf.stop_gradient(tf.transpose(pos_training_sample))  # 스탑 그레디언트?
            pos_training_samples.append(pos_training_sample)

            # neg_training_sample
            dist_of_movie_frequency = np.array(self.dist_of_movie_frequency)
            dist_of_movie_frequency = dist_of_movie_frequency / dist_of_movie_frequency.sum()
            indexes_of_neg_training_samples = np.random.choice(self.num_of_movies,
                                                               size=self.size_of_neg_training_sample,
                                                               p=dist_of_movie_frequency)
            start_index = indexes_of_neg_training_samples[0]
            neg_training_sample = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                           [1, self.size_of_movie_vector])
            for j in range(1, self.size_of_neg_training_sample):
                index = indexes_of_neg_training_samples[j]
                neg_training_sample = tf.concat([neg_training_sample,
                                                 tf.slice(self.embedded_movie_vectors, [index, 0],
                                                          [1, self.size_of_movie_vector])], axis=0)
            neg_training_sample = tf.stop_gradient(tf.transpose(neg_training_sample))  # 스탑 그레디언트?
            neg_training_samples.append(neg_training_sample)

        # cost1 : 로그0 주의!
        cost_of_negative_sampling_network = tf.to_double(tf.constant(0))
        for pos_training_sample in pos_training_samples:
            cost_of_negative_sampling_network = cost_of_negative_sampling_network + tf.reduce_mean(tf.log(
                tf.clip_by_value(tf.sigmoid(tf.matmul(self.embedded_movie_vectors, pos_training_sample)), 0.00001,
                                 0.99999)))

        for neg_training_sample in neg_training_samples:
            cost_of_negative_sampling_network = cost_of_negative_sampling_network + tf.reduce_mean(
                tf.log(tf.clip_by_value(1 - tf.sigmoid(tf.matmul(self.embedded_movie_vectors, neg_training_sample)),
                                        0.00001, 0.99999)))
        size_of_sample = self.size_of_pos_training_sample + self.size_of_neg_training_sample
        cost_of_negative_sampling_network = -(cost_of_negative_sampling_network / size_of_sample)  # 부호 체크!

        return cost_of_negative_sampling_network

    def recommend(self, list_of_movie_choices_by_user):
        my_connection = pymysql.connect(host='localhost', user=self.database_user, password=self.database_password,
                                        db=self.database_name, charset="utf8")
        my_cursor = my_connection.cursor()

        sql = "select movie_key " \
              "from movie " \
              "order by movie_key"
        my_cursor.execute(sql)
        total_movie_title_list = my_cursor.fetchall()

        list_of_movie_index_choices_by_user = []
        for i in range(len(list_of_movie_choices_by_user)):
            for j in range(len(total_movie_title_list)):
                if total_movie_title_list[j][0] == list_of_movie_choices_by_user[i]:
                    list_of_movie_index_choices_by_user.append(j)
                    break

        start_index = list_of_movie_index_choices_by_user[0]
        input_vector_of_user_network = tf.slice(self.embedded_movie_vectors, [start_index, 0],
                                                [1, self.size_of_movie_vector])
        for i in range(1, self.num_of_movie_vectors_to_make_input_vector_of_user_network):
            index = list_of_movie_index_choices_by_user[i]
            input_vector_of_user_network = tf.concat([input_vector_of_user_network,
                                                      tf.slice(self.embedded_movie_vectors, [index, 0],
                                                               [1, self.size_of_movie_vector])], axis=1)

        saver = tf.train.Saver()

        sess = tf.Session()
        saver.restore(sess, self.path_of_save)
        print("Model restored from %s" % (self.path_of_save))

        input_vector_of_user_network = sess.run(input_vector_of_user_network)

        predicted_score_matrix = sess.run(self.predicted_score_matrix, feed_dict={
            self.input_vectors_of_user_network_placeholder: input_vector_of_user_network})

        sql = "select title " \
              "from movie " \
              "order by movie_key"
        my_cursor.execute(sql)
        total_movie_title_list = my_cursor.fetchall()
        print(predicted_score_matrix)
        result = []
        len_of_predicted_score_list = len(predicted_score_matrix)
        for i in range(len_of_predicted_score_list):
            score = int(predicted_score_matrix[i])
            if score >= 8:
                result.append(total_movie_title_list[i])
        print(result)
        sess.close()

        my_connection.close()

    def visualize(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, self.path_of_save)
        print("Model restored from %s" % (self.path_of_save))

        embedded_movie_vectors = sess.run(self.embedded_movie_vectors)
        embedded_movie_vectors_tsne = TSNE(n_components=3, learning_rate=1000, init='pca').fit_transform(
            embedded_movie_vectors)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(len(embedded_movie_vectors_tsne)):
            x = embedded_movie_vectors_tsne[i][0]
            y = embedded_movie_vectors_tsne[i][1]
            z = embedded_movie_vectors_tsne[i][2]
            ax.scatter(x, y, z)
            ax.text(x, y, z, "{}".format(i))
        plt.show()

        input_vectors_of_user_network = sess.run(self.input_vectors_of_user_network)
        embedded_movie_vectors = sess.run(self.embedded_user_vectors, feed_dict={
            self.input_vectors_of_user_network_placeholder: input_vectors_of_user_network})
        embedded_movie_vectors_tsne = TSNE(n_components=3, learning_rate=1000, init='pca').fit_transform(
            embedded_movie_vectors)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(len(embedded_movie_vectors_tsne)):
            x = embedded_movie_vectors_tsne[i][0]
            y = embedded_movie_vectors_tsne[i][1]
            z = embedded_movie_vectors_tsne[i][2]
            ax.scatter(x, y, z)
            ax.text(x, y, z, "{}".format(i))
        plt.show()
