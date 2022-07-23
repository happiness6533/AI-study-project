function Z = projectData(X, U, K)

% K차원으로 축소
Z = X * U(:, 1:K);

end