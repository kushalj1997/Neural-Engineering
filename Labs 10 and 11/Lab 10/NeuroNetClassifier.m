% This is the implemented Neuron Network classification class whose
% structure consists of n - 1 sigmoid layer and 1 softmax layer.
% It usage mainly contains a constructor, a forward propagation function,
% a backward propagation function, and a cross-entropy error calculation
% function. This class is ideal for implementing neural networks for
% classification. 
classdef NeuroNetClassifier
    properties
        % Total number of layers (for example, for a 2-hidden-layer net,
        % the number will be 3 because there will be a softmax output 
        % layer). It is expected to be a integer
        num_layers
        
        % The dimension for each layer, it is expected to be an array
        % The first number is input number, second number is the first 
        % level neuron number, and so on. The last number is the output 
        % number
        dimensions_each_layer
        
        % It is a cell array, which stores all the coefficients 
        % that are necessary for computing output from input. It is 
        % initialized using the dimensions_each_layer variable, and the 
        % numbers are initialized to follow a Gaussian Distribution with 
        % mean 0 and variance of 1.
        % For each cell, it is a matrix, whose size is
        % (number_of_input + 1) * number_of_output. Number_of_inputs means
        % the input signal from previous level, and number_of_output means
        % the number of output signal.
        % Note: Here the "+1" term is to incorporate the constant term in 
        %       linear fitting
        coefficients
        
        % It is a cell array that has the same dimension as
        % coefficients. It stores the gradient square for each
        % coefficient that are back-propagated. This is used to adapatively
        % change the learning rate.
        sum_gradient_square
    end % properties
    methods
        % Constructor. It initializes the properties of this class.
        % Input: layers_number: an integer which specifies the amount of
        %                       total layers for this neural network
        %        neurons_per_layer: an array that specifies the dimension
        %                           for each layer. Please see the above
        %                           comments for dimensions_each_layer for
        %                           a detailed discrption.
        % Return: obj: The object that is initialized by this constructor
        function obj = NeuroNetClassifier(layers_number, neurons_per_layer)
            obj.num_layers = layers_number;
            obj.dimensions_each_layer = neurons_per_layer;
            for i = 1:obj.num_layers
                % Normal distribution works a lot better than uniform
                % distribution
                obj.coefficients{i} = ...
                    randn(obj.dimensions_each_layer(i) + 1, ...
                    obj.dimensions_each_layer(i + 1));
                % Initial sum should be 0
                obj.sum_gradient_square{i} = ...
                    zeros(obj.dimensions_each_layer(i) + 1, ...
                    obj.dimensions_each_layer(i + 1));
            end
        end % NeuroNetClassifier
        
        % Perform formward propagation for the given input.
        % Input: obj: The class instance.
        %        input: a matrix follows the size: 
        %               (number_of_input + 1) * sample_size
        % Output: a cell array where each cell stores corresponding layer's
        %         forward propagation result
        function result = ForwardPropagation(obj, input)
            result = cell(1,obj.num_layers);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %                         LAB 10, PART 2                       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % obj.num_layers has number of layers not counting input layer
            numHiddenLayers = obj.num_layers - 1;
            for i = 1:numHiddenLayers % <--you need to define this value
                % For each iteration, do the following:
                %   1. calculate output to the next layer
                % Grab the weights
                M = obj.coefficients{i};
                % Multiply input transpose and weights
                z = input' * M;
                % Transpose so the number of neurons in next layer are represented in rows
                z = z';
                % Apply the nonlinear transform (sigmoid 1 / (1 + e_i^(-x)) )
                for i = 1:size(z, 1)
                    for j = 1:size(z, 2)
                        z(i, j) = 1 ./ (1 + exp(-z(i, j)));
                    end
                end
                
                % Add the bias row
                N = size(input, 2);
                z = [z; ones(1, N)];
                %   2. save the output to results
                results{i} = z;
                %   3. pass the output on as the next layer's input
                input = z;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % The last layer is a softmax layer
            result{obj.num_layers} = obj.SoftmaxActivation(...
                obj.coefficients{obj.num_layers}, input);
        end % ForwardPropagation
        
        % Performs backpropagation to calculate the gradient for each
        % coefficient.
        % Input: obj: an instance of the class.
        %        input: a matrix of the input data, its size should be
        %               (number_input + 1) * sample_size
        %        target: a matrix of the training target that is stored
        %                using one-hot encoding. The size of the matrix is
        %                num_output * sample_size
        %        gamma_initial: the constant that is used to change
        %                       learning rate adaptively. During
        %                       implementation, it is randomly selected to
        %                       be 6.
        % Output: new_coeff: a cell array that stores the new coefficients.
        %         new_sum_gradient_square: a cell array that stores the new
        %                                  sum_gradient_square.
        % Note: The above two are needed because I think Matlab is doing
        %       pass by value, thus I cannot modify obj.coefficients and
        %       obj.sum_gradient_square from within the function. These two
        %       are used to update these two properties.
        function [new_coeff, new_sum_gradient_square] = BackPropagation(...
                obj, input, target, gamma_initial)
            forward_result = ForwardPropagation(obj, input);
            
            % The last layer is a softmax level (grad to abberviate
            % gradient)
            [coeff_grads{1}, grad_square_sums{1}, input_var_gradient] ... 
                = obj.SoftmaxBackProp(...
                    obj.coefficients{obj.num_layers}, ...
                    forward_result{obj.num_layers}, ...
                    forward_result{obj.num_layers - 1}, target);
            count = 2;
                
            % All the previous layers are sigmoid layers
            for i = obj.num_layers - 1:-1:2
                [coeff_grads{count}, grad_square_sums{count}, ...
                    input_var_gradient] = obj.SigmoidBackProp(...
                        obj.coefficients{i}, forward_result{i}, ...
                        forward_result{i - 1}, input_var_gradient);
                count = count + 1;
            end
            
            % First layer used input to get the first layer output
            [coeff_grads{count}, grad_square_sums{count}, ...
                input_var_gradient] = obj.SigmoidBackProp(...
                    obj.coefficients{1}, forward_result{1}, input, ...
                    input_var_gradient);
            
            % Calculates the sum_gradient_square and new coefficients
            for i = 1:obj.num_layers
                new_sum_gradient_square{i} = obj.sum_gradient_square{i} ...
                   + grad_square_sums{obj.num_layers + 1 - i};
                new_coeff{i} = obj.coefficients{i} - gamma_initial ./ ...
                   sqrt(new_sum_gradient_square{i}) .* ...
                   coeff_grads{obj.num_layers + 1 - i} ...
                   / size(input, 2);
            end
        end % BackPropagation
        
        % Calculate the error. The function calculates the cross-entropy
        % error. It will be useful for understanding the training
        % progress.
        % Input: obj: an instance of the class.
        %        input: a matrix of data that follows the size
        %               (num_input + 1) * sample_size
        %        target: a matrix of the training target that is stored
        %                using one-hot encoding. The size of the matrix is
        %                num_output * sample_size
        % Output: The calculated error that is stored in float (double)
        function error = CalculateError(obj, input, target)
            result = obj.ForwardPropagation(input);
            result = result{obj.num_layers};
            error = 0;
            for i = 1:size(input, 2)
                for j = 1:size(result, 1)
                    if (j == find(target(:, i) == 1))
                        error = error - log(result(j, i));
                    end
                end
            end
        end % CalculateError
    end % methods
    methods (Static)
        % Calculates the output for a softmax layer (together with the
        % corresponding linear combination for this layer)
        % Input: coeff: Linear combination's coefficients matrix, its size
        %               is input_number * output_number
        %        input: data matrix whose size is input_number *
        %               sample_size
        % Output: A matrix whose size is output_number * sample_size
        function result = SoftmaxActivation(coeff, input)
            % temp is the linear combination result
            temp = coeff' * input;
            result = zeros(size(coeff, 2), size(input, 2));
            % For each column, take the exponent, calculate column sum, and
            % divides by the sum to calculate the proabability for
            % obtaining this result
            for i = 1:size(temp, 2)
                exp_temp = exp(temp(:, i));
                total = sum(exp_temp);
                result(:, i) = exp_temp / total;
            end
        end % SoftmaxActivation
        
        % Calculates the gradients for a softmax layer.
        % Input: coeff: The linear combination's coefficients, its size
        %               is input_number * neuron_number
        %        output: The calculation output for the softmax layer given
        %                the input data, its size is 
        %                output_number * sample_number
        %        input: data matrix whose size is input_number *
        %               sample_size
        %        target: the training target matrix, its size should be
        %                output_number * sample_size
        % Output: coeff_gradient: the calculated sum gradient for each
        %                         coefficients given the input, it is a
        %                         matrix whose size is 
        %                         input_num * output_num
        %         coeff_gradient_square: the calculated sum gradient square
        %                                for each coefficients given the 
        %                                input, it is amatrix whose size is 
        %                                input_num * output_num. It is used
        %                                for adapt learning rate
        %         input_gradient: The gradient for each input corresponding
        %                         to each sample. Its size is 
        %                         input_number * sample_size. It is
        %                         propagated to calculate previous layer's
        %                         gradient.
        function [coeff_gradient,...
                  coeff_gradient_square,...
                  input_gradient] = SoftmaxBackProp(...
                    coeff, output, input, target)
            % Initialize output matrices
            coeff_gradient = zeros(size(coeff));
            coeff_gradient_square = zeros(size(coeff));
            input_gradient = zeros(size(input));
            for sample = 1:size(output, 2)
                for input_source = 1:size(input, 1)
                    for output_direction = 1:size(output, 1)
                        % Calculate the gradient for one coefficient in one
                        % sample, the derivation is inclueded in the
                        % paper together with it
                        gradient = output(output_direction, sample);
                        if (output_direction == find(target(:,...
                                                            sample) == 1))
                            gradient = gradient - 1;
                        end
                        gradient = gradient * input(input_source, sample);
                        coeff_gradient_square(input_source,...
                                              output_direction) =...
                            coeff_gradient_square(input_source,...
                                                  output_direction)...
                            + gradient ^ 2;
                        coeff_gradient(input_source,...
                                       output_direction) =...
                            coeff_gradient(input_source,...
                                           output_direction) + gradient;
                    end
                end
                for input_source = 1:size(input, 1)
                    % Calculate the gradient for each input source 
                    % corresponding to each sample
                    d_input = 0;
                    for output_direction = 1:size(output, 1)
                        temp = output(output_direction, sample);
                        if (output_direction == find(target(:,...
                                                            sample) == 1))
                            temp = temp - 1;
                        end
                        d_input = d_input + temp * coeff(input_source,...
                                                         output_direction);
                    end
                    input_gradient(input_source, sample) = d_input;
                end
            end
        end % SoftmaxBackProp
        
        % Calculates the gradients for a sigmoid layer.
        % Input: coeff: The linear combination's coefficients, its size
        %               is input_number * neuron_number
        %        output: The calculation output for the sigmoid layer given
        %                the input data, its size is 
        %                output_number * sample_number
        %        input: data matrix whose size is input_number *
        %               sample_size
        %        output_gradient: the gradient propagated from following 
        %                         layer, its size should be
        %                         output_number * sample_size
        % Output: coeff_gradient: the calculated sum gradient for each
        %                         coefficients given the input, it is a
        %                         matrix whose size is 
        %                         input_num * output_num
        %         coeff_gradient_square: the calculated sum gradient square
        %                                for each coefficients given the 
        %                                input, it is amatrix whose size is 
        %                                input_num * output_num. It is used
        %                                for adapt learning rate
        %         input_gradient: The gradient for each input corresponding
        %                         to each sample. Its size is 
        %                         input_number * sample_size. It is
        %                         propagated to calculate previous layer's
        %                         gradient.
        function [coeff_gradient,...
                  coeff_gradient_square,...
                  input_gradient] = SigmoidBackProp(...
                        coeff, output, input, output_gradint)
            % Initialize output matrices
            coeff_gradient = zeros(size(coeff));
            coeff_gradient_square = zeros(size(coeff));
            input_gradient = zeros(size(input));
            for sample = 1:size(output, 2)
                for input_source = 1:size(input, 1)
                    for output_direction = 1:size(output, 1) - 1
                        % Calculate the gradient for one coefficient in one
                        % sample, the derivation is inclueded in the
                        % paper together with it
                        delta = output_gradint(output_direction, sample)...
                            * output(output_direction, sample)...
                            * (1 - output(output_direction, sample))...
                            * input(input_source, sample);
                        coeff_gradient_square(input_source,...
                                              output_direction) = ...
                            coeff_gradient_square(input_source,...
                                                  output_direction)...
                            + delta ^ 2;
                        coeff_gradient(input_source, output_direction) =...
                            coeff_gradient(input_source,...
                            output_direction) + delta;
                    end
                end
                for input_source = 1:size(input, 1)
                    % Calculate the gradient for each input source 
                    % corresponding to each sample
                    d_y = 0;
                    for output_direction = 1:size(output, 1) - 1
                        d_y = d_y +...
                            output_gradint(output_direction, sample) *...
                            output(output_direction, sample) *...
                            (1 - output(output_direction, sample)) *...
                            coeff(input_source, output_direction);
                    end
                    input_gradient(input_source, sample) = d_y;
                end
            end
        end % SigmoidBackProp
    end % methods (Static)
end % NeuroNetClassifier