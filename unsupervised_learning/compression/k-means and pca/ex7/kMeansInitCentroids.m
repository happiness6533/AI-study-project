function centroids = kMeansInitCentroids(X, K)

% m개의 데이터 X에서 랜덤하게 추출한 K개의 데이터를 센터로 초기화
% randperm(10) : 1 - 10까지 랜덤하게 정렬된 1 * 10 행렬을 리턴한다
randidx = randperm(size(X, 1));
centroids = X(randidx(1:K), :);

end