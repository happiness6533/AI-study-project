% k-means
clear;
close all;
clc;


% 각 데이터를 가장 가까운 센터와 연결
fprintf('Finding closest centroids.\n\n');
load('ex7data2.mat');
K = 3;
initial_centroids = [3 3; 6 2; 8 5];
idx = findClosestCentroids(X, initial_centroids);
fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% 각 센터에 연결되어 있는 데이터들의 평균 구하기
% 평균 = 새로운 센터
fprintf('\nComputing centroids means.\n\n');
centroids = computeCentroids(X, idx, K);
fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% k-means 실행
fprintf('\nRunning K-Means clustering on example dataset.\n\n');
load('ex7data2.mat');
K = 3;
max_iters = 10;
initial_centroids = [3 3; 6 2; 8 5];
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% k-means 이미지 압축
% 256가지 색깔로 이루어진 이미지를 픽셀로 나누어 데이터화
% 센터 16개 생성 / 클러스터링
% 같은 센터에 해당하는 픽셀 데이터는 같은 값(센터 값)으로!
% 16가지 색깔(센터들)로만 이루어진 이미지 생성 / 압축 성공
fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');
load('bird_small.mat');

% 픽셀의 모든 값이 0 - 1 사이에 오도록 255로 나눈다
A = A / 255;
img_size = size(A);

% A = 픽셀 행 * 픽셀 열 * r/g/b >> A = 픽셀개수 * r/g/b 로 reshape
X = reshape(A, img_size(1) * img_size(2), 3);

K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);
fprintf('Program paused. Press enter to continue.\n');
pause;

printf('\nApplying K-Means to compress an image.\n\n');
idx = findClosestCentroids(X, centroids);

% a(b, :) = a행렬 중에서 b행렬의 각 원소값에 해당하는 행을 차례대로 출력한다
% a(b, :) = 열은 전부 출력한다
% 차례대로 = b행렬의 위에서 아래로 진행하면서 1열 클리어 >> 2열 클리어 >> ...
X_recovered = centroids(idx, :);

% 복구한 압축 이미지 3차원 행렬로 재생성
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% 오리지날 이미지 출력
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% 복구한 압축 이미지 출력
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
fprintf('Program paused. Press enter to continue.\n');
pause;