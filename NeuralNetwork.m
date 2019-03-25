% Juan Carlos ?ngeles Cer?n - A01271549
% Scalable Artificial Neural Network
% Tecnolog?as de Sistemas Inteligentes, Jan - May, 2019

clc; clear; close all;

data = (importdata('titanic.csv'));
test = (importdata('test.csv'));
data(:,6:11) = [];
data(:,1) = [];

input_dimensions = {4 2};
output_dimensions = {2 1};
activation = {'sigmoid' 'Sigmoid'};
cost_function = 'crossentropy';
epochs = 4;
learning_rate = 0.15;
regularization_rate = 0.0;
k_folds = 1;
save_nn_as = 'network.mat';

nn_architecture = nn_architecture(input_dimensions, output_dimensions, activation);

%%%%%% Esto dentro de K-folds
fold_accuracy = zeros(1, k_folds);
for k=1:k_folds
    fprintf('Fold %d\n',k);
    seed = 1;
    [train, validation] = get_train_test_set(data, .1);
    [X, Y] = format_data(train, 1, 0, 0);
    
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
            cost = get_cost(Y_hat, label, cost_function);
            epoch_cost = epoch_cost + cost;
            accuracy = get_accuracy(Y_hat, label);
            epoch_accuracy = epoch_accuracy + accuracy;

            grads_values = full_back_propagation(Y_hat, label, memory, params_values, nn_architecture);
            params_values = update_params(params_values, grads_values, nn_architecture, learning_rate, regularization_rate);
        end
        cost_history(i) = epoch_cost / length(X);
        accuracy_history(i) = epoch_accuracy / length(X);
        fprintf('Epoch: %d  Current cost: %f\n', i, cost)
    end
    %Train Function

    fold_accuracy(1,k) = accuracy_history(i);
end

disp('K-fold Accuracies');
disp(fold_accuracy)
%%%%%%%%

disp('Validation:')
[validation_x, validation_y] = format_data(validation, 1, 0, 0);
prediction_v = predict(validation_x, validation_y, params_values, nn_architecture);

disp('Test:')
test = test.data;
test(:,6:11) = [];
test(:,1) = [];
[test_x, test_y] = format_data(test, 1, 0, 1);
prediction_t = predict(test_x, test_y, params_values, nn_architecture);

subplot(2,1,1);
plot(cost_history);
xlabel('Time (Epochs)');
ylabel('Cost');
title('Cost performance vs time');
subplot(2,1,2);
plot(accuracy_history);
xlabel('Time (Epochs)');
ylabel('Accuracy (%)');
ylim([0 100])
title('Accuracy performance vs time');

nn = struct('nn_architecture', nn_architecture,'params_values', params_values);
save(save_nn_as, nn)
