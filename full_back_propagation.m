function [ grads_values ] = full_back_propagation( Y_hat, Y, memory, params_values, nn_architecture )
%FULL_BACK_PROPAGATION Computes one run of the back propagation algorithm
%   Inputs:
%       Y_hat              (vector)   Vector of outputs from the feed forward
%                                     algorithm
%       Y                  (vector)   Vector of expected output values.
%       params_values      (struct)   Structure of corresponding weights and biases for each layer.
%       nn_architecture    (struct)   Structure describing the architecture
%                                     of the neural network, with number of inputs and output neurons 
%                                     and activation fucntion per layer.
%                           
%   Outputs:                
%       grads_values       (struct)   Structure containing the gradient values to update the network 
%                                     weights. 

    grads_values = {};
    m = size(Y, 2);
    Y = reshape(Y, size(Y_hat));
    
    dA_prev = -((Y ./ Y_hat) - ((1 - Y) ./ (1 - Y_hat)));
    
    for i=length(nn_architecture):-1:1
        layer_idx = int2str(i);
        activation = nn_architecture(i).activation;
        
        dA_curr = dA_prev;
        
        A_prev = memory.(strcat('A', layer_idx - 1));
        Z_curr = memory.(strcat('Z', layer_idx));
        W_curr = params_values.(strcat('W', layer_idx));
        b_curr = params_values.(strcat('b', layer_idx));
        
        [dA_prev, dW_curr, db_curr] = single_layer_back_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation);
        
        [grads_values.(strcat('dW', layer_idx))] = dW_curr;
        [grads_values.(strcat('db', layer_idx))] = db_curr;
    end
end

