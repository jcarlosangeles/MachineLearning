function [ params_values ] = init_layers( nn_architecture, seed )
%INIT_LAYERS Randomly initializes the corresponding weights and biases for each layer
%            of the network architecture.
%   Inputs:
%       nn_architecture    (struct)   Structure describing the architecture of the neural network,                               
%                                     with number of inputs and output neurons and activation fucntion per layer.
%       seed               (int)      Seeds the random number generator using the integer seed so that random 
%                                     functions produce a predictable sequence of numbers.
%                           
%   Outputs:                
%       params_values      (struct)   Structure containing the weights and biases for each layer on the network.

    rng(seed);
    number_of_layers = length(nn_architecture);
    params_values = {};
    
    for i=1:number_of_layers
        layer_idx = int2str(i);        
        layer_input_size = nn_architecture(i).input_dimension;
        layer_output_size = nn_architecture(i).output_dimension;
        
        %Delete this and uncomment below once it runs smooth
%         [params_values.(strcat('W', layer_idx))] = ones(layer_output_size, layer_input_size);
%         [params_values.(strcat('b', layer_idx))] = ones(layer_output_size, 1);
        
        %Assign random values to weights and biases
        [params_values.(strcat('W', layer_idx))] = rand(layer_output_size, layer_input_size) * .1;
        [params_values.(strcat('b', layer_idx))] = rand(layer_output_size, 1) * .1;
    end 
%     params_values = struct('W1', [.15 .20; .25 .30], 'b1', [.35; .35], 'W2', [.40 .45; .50 .55], 'b2', [.60;.60]);
end
 