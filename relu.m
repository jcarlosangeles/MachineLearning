function [ eval ] = relu( Z )
%RELU Computes the ReLU activation function of a set of neurons
%   Inputs:
%       Z       (vector)   Vector of weighted sums of inputs or
%                                 activation values.
%                           
%   Outputs:                
%       eval    (vactor)   Evaluation of the ReLU function.

    eval = max(0, Z);
end

