function [ dZ ] = relu_backward( dA, Z )
%RELU_BACKWARD Computes the derivative of the relu function used in backpropagation
%   Inputs:
%       dA    (vector)   Vector of output errors of the current layer.
%       Z     (vector)   Vector of weighted sums calculated on forward
%                        propagation.
%                           
%   Outputs:                
%       dZ    (struct)   Derivative of the evaluation of the ReLU function.

    dZ = dA;
    dZ(Z <= 0) = 0;
end

