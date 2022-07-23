- natural language processing
	- similarity
		- 자카드 유사도 = 두 문장의 토큰들의 교집합의 원소의 개수 / 합집합의 원소의 개수
		- 코사인 유사도 = 벡터끼리의 코싸인
		- 유클리디언 유사도 = 벡터끼리의 거리
		- 멘하탄 유사도 = 벡터끼리 차의 절대값의 모든 원소의 합

	- EDA = 탐색적 데이터 분석 = exploratory data analysis
		- 데이터 전처리 > 정제된 전처리 > 탐색적 자료 분석(= EDA) > 모델 > 사용 |_________________________________|

		- 데이터 다운 > EDA(분석) > 데이터 정제(HTML 및 문장부호 제거, 불용어 제거, 단어 최대 길이 설정, 단어 패딩, 벡터 표상화) > 모델링

	- word imbedding
		- one hot encoding
			- 희소성 문제 + 관계 표현력 문제
		- 분포 가설 제안 = 벡터의 크기를 작게 하면 비슷한 벡터끼리는 비슷한 공간에 존재할 것이다
		- 카운트 기반
			- 등장횟수를 행렬로 수치화하고(동시 출현 행렬 )그것을 이용해서 단어로 변환
			- svd lsa hal pca 사용
		- 예측 기반
			- word2vec
				- cbow
					- 주변의 2 * window_size 개의 단어들을 원핫으로 만든 후 입력으로 사용
					- w를 곱해서 n차원 벡터 여러개 생성
					- 평균값을 계산한 후 다시 w를 곱해서 원핫을 타겟으로 하는 벡터 출력
					- 로스 최적화
				- skip-gram
			- nnlm(neural network language model)
			- rnnlm(recurrent neuralnetwork language model)
		- 카운트 + 예측: Glove

	- sentence classification
		- bag-of-word n-gram tf-idf
		- seq2seq
		- conv

	- sentence clustering
		- seq2seq
		- xg 부스트 추ㅜ ㅡ민스

	- sentence generation
		- seq2seq-with-attention
		- transformer
			- 멀티 헤드 어텐션
				- 스케일 내적 어텐션
					- query key value
					- attention(q, k, v) = softmax((q와 k 내적) / 루트 d) * v
					- 위에서 내적 후 루트d로 나누는 과정을 스케일이라고 한다
					- 각 단어에 해당하는 스코어를 각 v에 곱한다
					- 위의 계산으로 나온 모든 벡터들의 가중합을 query에 대한 key의 컨텍스트 벡터로 한다
					- 실제 코드에서는 q k v가 모두 임베딩 벡터들의 모임으로 이루어진 문장 행렬이다
				- 순방향 어텐션 마스크
					- 현재 예측하고자 하는 단어 뒤의 단어는 예측하지 않도록 어텐션 스코어에 삼각 마스크를 씨워서 어텐션 연산을 하는 기법이다
				- 멀티 헤드 어텐션
					- q k v에 대해 스케일 내적 어텐션을 여러개를 수행한 다음 concat하고 linear를 통과시키는 과정
			- 서브시퀀트 마스크 어텐션
			- 포지션 와이즈 피드 포워드 네트워크
			- 레지듀얼 커넥션

	- machine comprehension
		- BABI 데이터셋을 통해 텍스트, 질문을 주고 답변을 하는 예시 존재
			- 페북 팀에서 20가지 부류의 질문 내용으로 구성
			- 시간 순서대로 나열된 텍스트와 그에 대한 질문 => 응답하는 형태
			- 사람의 성능을 이김
		- SQuAD
			- 46주제 10만개질문
		- VQA
			- 이미지를 주고 텍스트로 질문을 던져서 답변

	- language model
		- bert
		- elmo
		- gpt

	- 정리 예정중
		- 사이킷런?
		- 토크나이징 도구(영 한)
		- 넘파이 판다스 맷플랏 re
		- 캐글