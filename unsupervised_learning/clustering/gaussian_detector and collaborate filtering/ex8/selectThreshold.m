function [bestEpsilon bestF1] = selectThreshold(yval, pval)

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
  m = size(pval, 1);  
  validationPval = zeros(size(pval));
  for i = 1:m
    if pval(i, :) < epsilon
      validationPval(i, :) = 1;
    endif
  endfor
  
  truePos = sum((yval == 1) & (validationPval == 1));
  predictedPos = sum(validationPval == 1);
  actualPos = sum(yval == 1);

  precision = truePos / predictedPos;
  recall = truePos / actualPos;
  
  F1 = (2 * precision * recall) / (precision + recall);
  
  if F1 > bestF1
    bestF1 = F1;
    bestEpsilon = epsilon;
  end
end

end