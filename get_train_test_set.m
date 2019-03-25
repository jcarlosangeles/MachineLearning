function [ train_set, test_set ] = get_train_test_set( dataset, size )
%GET_TRAIN_TEST_SET Randomly generates a training and test set given a
%                   single dataset and the desired size for the test set.
%                   This function must be used before separating features
%                   and labels.
%   Inputs:
%       dataset      (matrix)       Matrix of training data to be splitted.
%       size         (float/int)    Desired size of the test set. If size
%                                   is >1, the size is interpreted as number of rows. If the size is
%                                   <1, the size is interpreted as a percentage of the total length of
%                                   the dataset
%                           
%   Outputs:                
%       train_set    (matrix)       (M - size x N) Matrix of training data
%       test_set     (matrix)       (size x N) matrix of test data.

    train_set = dataset;
    start_idx = round(1 + (length(dataset)-1) .* rand(1,1));
    
    if size < 1
       size = floor(length(dataset) * size); 
    end
    
    if start_idx+size <= length(dataset)
        test_set = train_set(start_idx:start_idx + size - 1,:);
        train_set(start_idx:start_idx + size,:) = [];
    else
        rows_left = start_idx + size - length(dataset);
        test_set = train_set(start_idx:end,:);
        train_set(start_idx:end,:) = [];
        test_set = [test_set; train_set(1:rows_left - 1,:)];
        train_set(1:rows_left,:) = [];
    end

end

