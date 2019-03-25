function [ eval ] = sigmoid( z )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    eval = 1./(1+exp(-z));
end
