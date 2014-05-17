function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

paramsToTry = [0.01 0.03 0.1 0.3 1 3 10 30]'
results = zeros(power(size(paramsToTry, 1), 2), 3);

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

doSearch = 0;
if(doSearch == 1)

n = 1;
sizeParams = size(paramsToTry, 1);

for c = 1 : sizeParams 

   for s = 1 : sizeParams 
       
       gsFunc = @(x1, x2) gaussianKernel(x1, x2, paramsToTry(s));

       model = svmTrain(X, y, paramsToTry(c), gsFunc);

       predictions = svmPredict(model, Xval);

       predError = mean(double(predictions ~= yval));
   
       aResult = [paramsToTry(c) paramsToTry(s) predError]
       results(n, :) = aResult;
       n = n + 1;

   endfor

endfor

[W, IW] = min(results(:, 3))

C = results(IW, 1)
sigma = results(IW, 2)

else

    C = 1.0
    sigma = 0.1

endif
% =========================================================================

end