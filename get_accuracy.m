function [ accuracy ] = get_accuracy( Y_hat, Y )
%GET_ACCURACY Computes the percentual accuracy between two vectors
%   Inputs:
%       Y_hat       (vector)   Vector of outputs from the feed forward
%                                     algorithm
%       Y           (vector)   Vector of expected output values.
%                           
%   Outputs:                
%       Accuracy    (float)    Percentual accuracy between the two
%                              vectors

    Y = reshape(Y, size(Y_hat));
    Y_hat(Y_hat > .5) = 1;
    Y_hat(Y_hat <= .5) = 0;
    
    accuracy = (Y_hat == Y);
    accuracy = mean(all(accuracy, 1)) * 100;
end

