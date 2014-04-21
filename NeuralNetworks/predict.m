function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
sizeTheta1 = size(Theta1);
sizeTheta2 = size(Theta2);
sizeX = size(X);
sizeP = size(p);

X1 = [ones(m, 1) X];
sizeX1 = size(X1);

% Hidden Layer
a2 = sigmoid(transpose(Theta1 * transpose(X1)));
sizeA2pre = size(a2);

% Add the bias value to the a2 output
a2 = [ones(sizeA2pre(1), 1) a2];
sizeA2 = size(a2);

% Output layer
a3 = sigmoid(transpose(Theta2 * transpose(a2)));
sizeA3 = size(a3);

% Assign output prediction
[x, p] = max(a3, [], 2);

sizeP = size(p);
% =========================================================================


end
