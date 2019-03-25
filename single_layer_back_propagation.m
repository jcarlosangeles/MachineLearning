function [ dA_prev, dW_curr, db_curr ] = single_layer_back_propagation( dA_curr, W_curr, b_curr, Z_curr, A_prev, activation )
%SINGLE_LAYER_BACK_PROPAGATION Computes one layer of the back propagation algorithm
%   Inputs:
%       dA_curr       (vector)    Delta of the current activation values.
%       W_curr        (matrix)    Set of weights of the curre1nt layer.
%       b_curr        (vector)    Set of biases of the current layer.
%       Z_curr        (matrix)    Set of weighted sums of the current layer.
%       A_prev        (vector)    Set of activation values of the previous layer
%       activation    (string)    Activation function of the current layer
%                           
%   Outputs:                
%       dA_prev       (vector)    Set of activation deltas of the previous layer.  
%       dW_curr       (matrix)    Matrix of deltas of the current layer weights.
%       db_curr       (vector)    Vector of deltas of the current layer biases.

    m = size(A_prev,2);
    
    if (strcmpi(activation, 'sigmoid'))
        dZ_curr = sigmoid_backward(dA_curr, Z_curr);
    elseif (strcmpi(activation, 'relu'))
        dZ_curr = relu_backward(dA_curr, Z_curr);
    else
        error('Unknown activation function');
    end
   
    dW_curr = (dZ_curr * A_prev) / m;
    db_curr = sum(dZ_curr, 2) / m;
    dA_prev = W_curr.' * dZ_curr;
       
end

