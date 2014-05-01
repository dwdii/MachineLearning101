function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 

...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
sizeY = size(y);
sizeT1 = size(Theta1);
sizeT2 = size(Theta2);

% Add the bias vector to the training samples
X1 = [ones(size(X, 1), 1) X];
sizeX1 = size(X1);

D1 = zeros(hidden_layer_size, size(X1, 2));
D2 = zeros(num_labels, hidden_layer_size + 1); %sizeT2(2) - 1, 1);


% Loop through the samples to sum up the cost
mSum = 0;
for i = 1 : m

    % z = h(X)
    Z2 = Theta1 * X1(i, :)';
    sizeZ2 = size(Z2);

    % a2 = g(z)
    a2 = transpose(sigmoid(Z2));
    sizea2 = size(a2);

    % add bias 
    a21 = [ones(size(a2, 1), 1) a2];
    sizea21 = size(a21);

    % z = h(a2)
    Z3 = Theta2 * transpose(a21);
    sizeZ3 = size(Z3);

    a3 = sigmoid(Z3);
    sizea3 = size(a3);

    a31 = transpose(a3);
    sizea31 = size(a31);

    yk = zeros(num_labels, 1);
    yk(y(i)) = 1;
    
    for k = 1 : num_labels
    
       % Cost summation
       p1 = yk(k) * log(a31(k));
       sizeP1 = size(p1);
    
       p2 = (1 - yk(k)) * log(1 - a31(k));
       sizeP2 = size(p2);
      
       mSum += p1 + p2;
       
    endfor 
    
    % Backprop
    delta3 = a3 - yk;
    size_delta3 = size(delta3);

    % Backprop continued
    delta2 = (transpose(Theta2(:, 2:end)) * delta3) .* sigmoidGradient(Z2);
    size_delta2 = size(delta2);

    D2 = D2 + (delta3 * a21);
    sizeD2 = size(D2);
    
    D1 = D1 + (delta2 * X1(i, :));
    sizeD1 = size(D1);
endfor

Theta2_grad = (1/m) * D2 + ((lambda/m) * Theta2);
Theta2_grad(:, 1) = (1/m) * D2(:, 1);
sizeT2_grad = size(Theta2_grad);

Theta1_grad = (1/m) * D1 + ((lambda/m) * Theta1);
Theta1_grad(:, 1) = (1/m) * D1(:, 1);
sizeT1_grad = size(Theta1_grad);

% Compute Cost (pre-regularization)
J1 = (-1 / m) * mSum;

% Apply regularization to Cost
T1sq = 0;
T2sq = 0;

for j = 1 : sizeT1(1)
    for(k = 2 : sizeT1(2))
        T1sq += power(Theta1(j, k), 2);
    endfor
endfor

for j = 1 : sizeT2(1)
    for(k = 2 : sizeT2(2))
        T2sq += power(Theta2(j, k), 2);
    endfor
endfor

T1sq;
T2sq;
J1;

J = J1 + ((lambda / (2 * m)) * ( T1sq + T2sq));




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end