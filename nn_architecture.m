function [ nn_architecture ] = nn_architecture( input_dimension, output_dimension, activation )
%NN_ARCHITECTURE Creates a structure with input and output dimensions for
%                each layer, as well as their activation functions
%   Inputs:
%       input_dimension     (vector)   Vector of input neurons on each layer.
%       output_dimension    (vector)   Vector of output neurons on each layer.
%       activation          (vector)   Vector of strings, each element representing the activation
%                                      function to be used in that specific layer. 'sigmoid' or 'relu'
%                           
%   Outputs:                
%       nn_architecture     (struct)   Structure describing the architecture
%                                      of the neural network, with number of inputs and output neurons 
%                                      and activation fucntion per layer.
    field1 = 'input_dimension';
    field2 = 'output_dimension';
    field3 = 'activation';
    
    value1 = input_dimension;
    value2 = output_dimension;
    value3 = activation;
    
    nn_architecture = struct(field1, value1, field2, value2, field3, value3);
end

