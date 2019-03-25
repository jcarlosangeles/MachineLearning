function [ params_values, cost_history, accuracy_history ] = train( X, Y, nn_architecture, epochs, learning_rate )
%TRAIN Trains a neural network for a set number of epochs
%   Inputs:
%       X                   (matrix)    Matrix of features.
%       Y                   (matrix)    Matrix of labels.
%       nn_architecture     (struct)    Structure describing the architecture
%                                       of the neural network, with number of inputs and output neurons 
%                                       and activation fucntion per layer.
%       epochs              (integer)   Number of training cycles.
%       learning_rate       (float)     Learning rate for gradient descent.
%                           
%   Outputs:                
%       params_values       (struct)    Structure containing the last updated weights and biases.
%       cost_history        (vector)    Vector of cost value throughout each epoch.
%       accuracy_history    (vector)    vector of accuracy performance throughout each epoch.

    %Train Function
    params_values = init_layers(nn_architecture, seed);
    cost_history = zeros(1,epochs);
    accuracy_history = zeros(1,epochs);
    
    for i=1:epochs
        epoch_cost = 0;
        epoch_accuracy = 0;
        for j=1:length(X)
            sample = X(j,:);
            label = Y(j,:);
            [Y_hat, memory] = full_forward_propagation(sample, params_values, nn_architecture);
            cost = get_cost(Y_hat, label);
            epoch_cost = epoch_cost + cost;
            accuracy = get_accuracy(Y_hat, label);
            epoch_accuracy = epoch_accuracy + accuracy;

            grads_values = full_back_propagation(Y_hat, label, memory, params_values, nn_architecture);
            params_values = update_params(params_values, grads_values, nn_architecture, learning_rate);
        end
        cost_history(i) = epoch_cost / length(X);
        accuracy_history(i) = epoch_accuracy / length(X);
        fprintf('Epoch: %d  Current cost: %f\n', i, cost)
    end
    disp(params_values)
    disp(cost_history)
    disp(accuracy_history)
end

