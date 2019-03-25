function [ dZ ] = sigmoid_backward( dA, Z )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    sig = sigmoid(Z);
    dZ = dA .* sig .* (1 - sig);
end

