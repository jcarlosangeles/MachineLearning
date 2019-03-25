function [ normalized ] = normalize_data( samples )
%NORMALIZE Normalizes the sample matrix column-wise to values ranging from 0 to 1
%   Inputs:
%       samples    (matrix) Matrix containing training data.
%
%   Outputs:
%       normalized (matrix) Normalized training data matrix.

    sz = size(samples);
    normalized = zeros(sz);
    for i=1:sz(2)
        col = samples(:,i);
        col = (col - min(col)) / (max(col) - min(col));
        normalized(:,i) = col;  
    end
end

