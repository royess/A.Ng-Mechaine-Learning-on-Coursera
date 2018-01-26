function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
err_min = 1; % magic number...

for i = 1:8
  for j = 1:8
    fprintf('#%d\n',8 * (i - 1) + j);
    C_temp = vals(i);
    sigma_temp = vals(j);
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    preds = svmPredict(model ,Xval);
    err = mean(double(preds ~= yval));
    fprintf('error:\t%f\n',err);
    if (err < err_min)
      err_min = err;
      C = C_temp;
      sigma = sigma_temp;
    end
  end
end








% =========================================================================

end
