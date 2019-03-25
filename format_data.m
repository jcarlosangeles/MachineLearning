function [ X, Y ] = format_data( dataset, normalize, one_hot, remove_headers )
%FORMAT_DATA Preprocess data to be ready to use as input
%   Inputs:
%       dataset           (matrix)   Matrix containing all training data and output 
%                                    classes (these must be on the last column)
%       normalize         (boolean)  Set this argument to true to normalize the
%                                    dataset column-wise.
%       one-hot           (boolean)  Set this argument to true if there is more than
%                                    two output classes. This generates the one-hot 
%                                    encoded output vector.
%       remove_headers    (boolean)  Set to true to remove headers from the .csv file.
%       
%   Outputs:
%       X                 (matrix)   Matrix of features 
%       Y                 (matrix)   Matrix of labels

    if remove_headers == true
        dataset = dataset(2:end,:);
    end
    
    Y = dataset(:,end);
    X = dataset(:,1:end-1);
    
    if one_hot == true
        classes = unique(Y);
        Y_hot = zeros(length(Y), length(classes));
        for i=1:length(classes)
            Y_hot(:,i) = (Y == classes(i));
        end
         Y = Y_hot;
    end
    
    if normalize == true
        X = normalize_data(X);
    end
end

