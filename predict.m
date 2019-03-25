function [ prediction ] = predict( test_x, test_y, params_values, nn_architecture )
%PREDICT Applies feed forward propagation to predict an output once the
%        network is trained.
%   Inputs:
%       test_x             (matrix)   Feature data of the test set.
%       test_y             (matrix)   Label data of the test set.
%       params_values      (struct)   Structure of corresponding weights and biases for each layer.
%       nn_architecture    (struct)   Structure describing the architecture
%                                     of the neural network, with number of inputs and output neurons 
%                                     and activation fucntion per layer.
%                           
%   Outputs:                
%       prediction         (vector)   Vector of class prediction for each
%                                     sample on the training data

    prediction = zeros(length(test_y), size(test_y, 2));
    
    for i=1:length(test_y)
        sample = test_x(i,:);
        
        Y_hat = round(full_forward_propagation(sample, params_values, nn_architecture));
        prediction(i,:) = Y_hat.';
%         fprintf('Actual: %d, Predicted: %d\n', test_y(i,:), prediction(i,:))
    end
    
    accuracy = (prediction == test_y);
    accuracy = mean(all(accuracy, 2)) * 100;
    fprintf('Accuracy on set: %f\n', accuracy);    
end

