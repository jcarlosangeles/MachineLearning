function [ params_values ] = update_params(params_values, grads_values, nn_architecture, learning_rate, regularization_rate)
%UPDATE_PARAMS Updates weights and biases using gradient descent
%   Inputs:
%       params_values       (struct)    Structure containing the last updated weights and biases.
%       grads_values        (struct)    Structure containing the gradient values to update the network 
%                                       weights.
%       nn_architecture     (struct)    Structure describing the architecture
%                                       of the neural network, with number of inputs and output neurons 
%                                       and activation fucntion per layer.
%       learning_rate       (float)     Learning rate for gradient descent.
%                           
%   Outputs:                
%       params_values       (struct)    Structure containing the last updated weights and biases.

    for i=1:length(nn_architecture)
        layer_idx = int2str(i);
        %So annoying to write and read. Hate you, Matlab -.-
%         disp(X)
%         disp(grads_values.(strcat('dW', layer_idx)))
        params_values.(strcat('W', layer_idx)) = params_values.(strcat('W', layer_idx)) - learning_rate * grads_values.(strcat('dW', layer_idx)) - learning_rate * (regularization_rate * params_values.(strcat('W', layer_idx)));
%         disp(params_values)
        params_values.(strcat('b', layer_idx)) = params_values.(strcat('b', layer_idx)) - learning_rate * grads_values.(strcat('db', layer_idx)) - learning_rate * (regularization_rate * params_values.(strcat('b', layer_idx)));
    end
    
%     new_params_values = params_values;
end