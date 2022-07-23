from supervised_learning_ml_training_server.projects.recommend_movie import recommend

naver_id = input("movie_data_agent를 위한 네이버 아이디를 입력하세요(보안을 위해 테스트용 아이디를 권장합니다) : ")
naver_password = input("movie_data_agent를 위한 네이버 비밀번호를 입력하세요(보안을 위해 테스트용 비밀번호를 권장합니다) : ")
database_user = input("movie_data_agent를 위한 데이터베이스 사용자 아이디를 입력하세요 : ")
database_password = input("movie_data_agent를 위한 데이터베이스 사용자 비밀번호를 입력하세요 : ")
database_name = input("movie_data_agent를 위한 데이터베이스 이름을 입력하세요 : ")
movie_data_agent = data.MovieDataAgent(100, 5000, naver_id, naver_password, database_user, database_password,
                                       database_name)

get_data = input("데이터 획득(y/n) : ")
if get_data == "y":
    reset = input("데이터 베이스 리셋(y/n) : ")
    if reset == "y":
        movie_data_agent.run(start=1, num_of_data=10, num_of_loop=100000, bool_of_reset_database=True)
    else:
        movie_data_agent.run(start=1, num_of_data=10, num_of_loop=100000, bool_of_reset_database=False)
print()

num_of_show_movies = int(input("선호작을 고르기 위해 필요한 샘플 영화의 개수를 입력하세요 : "))
num_of_choice = int(input("선택할 선호작의 개수를 입력하세요 : "))
num_of_recommend = int(input("추천받고 싶은 영화의 개수를 선택하세요 : "))
print()
movies_of_user_list = movie_data_agent.show_data_to_user_for_choice(num_of_show_movies=num_of_show_movies,
                                                                    num_of_choice=num_of_choice)

print("데이터 베이스에 의한 추천입니다")
print("loading...")
movie_data_agent.recommend_by_database(movies_of_user_list, num_of_recommend=num_of_recommend)
print()

similarity = input("similarity 데이터 초기화(y/n) : ")
if similarity == "y":
    movie_data_agent.create_database3()
print("ai의 훈련을 위한 데이터를 생성중입니다")
print("loading...")
len_of_total_movie_key_list, len_of_total_user_key_list, dists_of_movie_similarity, dist_of_movie_frequency, dists_of_movie_choice_by_user = movie_data_agent.get_data_for_recommend_agent()
print(len_of_total_movie_key_list)
print(len_of_total_user_key_list)
print(dists_of_movie_similarity)
print(dist_of_movie_frequency)
print(dists_of_movie_choice_by_user)
print()

print("ai를 생성중입니다")
print("loading...")
recommend_agent = recommend.RecommendAgent(
    num_of_movies=len_of_total_movie_key_list,
    size_of_movie_vector=64,
    dists_of_movie_similarity=dists_of_movie_similarity,
    size_of_pos_training_sample=5,
    dist_of_movie_frequency=dist_of_movie_frequency,
    size_of_neg_training_sample=15,
    learning_rate=0.0005,
    dists_of_movie_choice_by_user=dists_of_movie_choice_by_user,
    num_of_movie_vectors_to_make_input_vector_of_user_network=num_of_choice,
    num_of_users=len_of_total_user_key_list,
    score_matrix=dists_of_movie_choice_by_user,
    const_lambda=0.0005,
    alpha=0.4,
    beta=0.6,
    num_of_training=5000,
    path_of_save="save/model.save",
    database_user=database_user,
    database_password=database_password,
    database_name=database_name
)

training = input("학습(y/n) : ")
if training == "y":
    recommend_agent.learn()
    print()

print("visualize...")
recommend_agent.visualize()
print()

print("ai에 의한 추천입니다")
recommend_agent.recommend(movies_of_user_list)
print()
