function centroids = computeCentroids(X, idx, K)

m = size(X, 1);
n = size(X, 2);

% 센터 / 센터카운트 생성
centroids = zeros(K, n);
centroidsCount = zeros(K, 1);

% 같은 센터를 가지는 데이터들의 합
for i = 1:K
  for j = 1:m
    if idx(j, :) == i
      centroids(i, :) += X(j, :);
      centroidsCount(i, :) += 1;
    endif
  endfor
endfor

% 합 / 카운트 = 평균
centroids = centroids ./ centroidsCount;

end