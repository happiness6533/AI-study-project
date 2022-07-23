function X_rec = recoverData(Z, U, K)

% 오리지널 차원으로 복구
% U벡터 = 아이겐벡터, 따라서 역행렬 = transpose!
X_rec = Z * U(:, 1:K)';

end