function [U, S] = pca(X)

% 공분산(시그마)의 아이겐벡터 + 아이겐벡터값 구하기
m = size(X, 1);
n = size(X, 2);
sigma = (X' * X) / m;
[U, S, V] = svd(sigma);

end