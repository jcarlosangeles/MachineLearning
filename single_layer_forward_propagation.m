    function [ A_curr, Z_curr ] = single_layer_forward_propagation( A_prev,  W_curr, b_curr, activation )
%SINGLE_LAYER_FORWARD_PROPAGATION Computes one layer of the feed fprward propagation algorithm
%   Inputs:
%       A_prev        (vector)    Vector of activation values of the previous layer.
%       W_curr        (matrix)    Set of weights of the curre1nt layer.
%       b_curr        (vector)    Set of biases of the current layer.
%       activation    (string)    Activation function of the current layer.
%                           
%   Outputs:                
%       A_curr        (vector)    Vector of activation values of the current layer.  
%       Z_curr        (vector)    Vector of weighted sums for each neuron on the current layer. 

    Z_curr = (W_curr * A_prev.') + b_curr;

    if strcmpi(activation, 'sigmoid') == 1
        A_curr = sigmoid(Z_curr);
    elseif strcmpi(activation, 'relu') == 1
        A_curr = relu(Z_curr);
    else 
        error('Unknown activation function');
    end

end

