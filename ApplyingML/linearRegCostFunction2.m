function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y);  % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
sizeTheta = size(theta);
sizeX = size(X);

h0 = X * theta;

err = h0 - y;

sqrErr = power(err, 2);

sumSqrErr = sum(sqrErr);

T2 = power(theta, 2);

sumT2 = sum(T2(2:end));

J = (1/ (2 * m)) * sumSqrErr + ( (lambda / ( 2 * m) ) * sumT2 );

% Calculate gradient for this J
for j = 2 : sizeTheta(1)
    errX = err .* X(:, j);
    sumG =  sum( errX );
    grad(j) = (1/m) * sumG + ( (lambda/m) * theta(j) );
endfor

grad(1) = (1/m) * sum( err .* X(:, 1) );

% =========================================================================

grad = grad(:);

end
