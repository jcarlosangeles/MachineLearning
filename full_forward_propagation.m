function [ A_curr, memory ] = full_forward_propagation( x, params_values, nn_architecture )
%FULL_FORWARD_PROPAGATION Computes one run of the back propagation algorithm
%   Inputs:
%       x                  (vector)  Vector of features corresponding to
%                                    one row of the dataset.
%       params_values      (struct)  Structure of corresponding weights and biases for each layer.
%       nn_architecture    (struct)  Structure describing the architecture
%                                     of the neural network, with number of inputs and output neurons 
%                                     and activation fucntion per layer.
%       
%   Outputs:
%       A_curr             (vector)   Vector of activation values on the output layer.
%       memory             (struct)   Structure of activation values and
%                                     inputs for each neuron on each layer.

    memory = {};
    A_curr = x;
    
    for i=1:length(nn_architecture)
        layer_idx = int2str(i);
        A_prev = reshape(A_curr, 1, []);
        
        W_curr = params_values.(strcat('W', layer_idx));
        b_curr = params_values.(strcat('b', layer_idx));
        activation_curr = nn_architecture(i).activation;     
        
        [A_curr, Z_curr] = single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_curr);
        [memory.(strcat('A', layer_idx-1))] = A_prev;
        [memory.(strcat('Z', layer_idx))] = Z_curr; 
    end
end

