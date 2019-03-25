function [ cost ] = get_cost( Y_hat, Y, type )
%GET_COST Computes the total cost between two vectors
%   Inputs:
%       Y_hat    (vector)   Vector of outputs from the feed forward
%                           algorithm
%       Y        (vector)   Vector of expected output values.
%       type     (string)   Determines the cost function to analize the
%                           total cost. 'MeanSquare' or 'CrossEntropy'.
%                           
%   Outputs:                
%       cost     (float)    Total cost among all output instances.

    m = size(Y_hat, 2);
    
    if strcmpi(type, 'meansquare')
        cost = (1 / m) * (Y - Y_hat.').^2;
    elseif strcmpi(type, 'crossentropy')
        cost = (1 / m) * (sum(-Y.' .* log(Y_hat) - ((1 - Y.') .* log(1 - Y_hat))));
    else
        error('Non-supported cost function')
    end
    
    cost = sum(cost);
end

