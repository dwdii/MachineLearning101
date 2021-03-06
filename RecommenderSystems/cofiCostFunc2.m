function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
num_users;
num_movies;
num_features;
sizeTheta = size(Theta);
sizeX = size(X);
sizeThetaj = size(Theta(1, :));
sizeXi = size(X(1, :));

% 2.2.1 Cost

ttX = Theta * transpose(X);

ttXy2 = power(transpose(ttX) - Y, 2);

vecSumSqrErr = sum(sum(R .* ttXy2));

J = (1/2) * vecSumSqrErr;


% 2.2.2 Gradient


for i = 1 : num_movies

    runningSumX = 0;
    for  j = 1 : num_users
        
        if(R(i, j) == 1)
        
            tX = Theta(j, :) * transpose(X(i, :));
            size_tX = size(tX);
            tXyDiff = tX - Y(i,j);
            runningSumX += (tXyDiff) * transpose(Theta(j, :));
        
        endif
    
    endfor 
    
    X_grad(i, :) = runningSumX;

endfor

for j = 1 : num_users

    runningSumT = 0;
    for  i = 1 : num_movies
        
        if(R(i, j) == 1)
        
            tX = Theta(j, :) * transpose(X(i, :));
            tXyDiff = tX - Y(i,j);
        
            runningSumT += (tXyDiff) * transpose(X(i, :));
        
        endif
    
    endfor 
    
    Theta_grad(j, :) = runningSumT;

endfor

R;
X_grad;
Theta_grad;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
