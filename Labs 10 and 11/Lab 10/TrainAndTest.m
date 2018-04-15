function [test_accuracy, train_accuracy] = TrainAndTest(...
    all_data, all_label,all_result_matrix, n_fold)
    % The driving program for training neural network and use the trained
    % network for testing
    % Input: all_data: All the collected data. Its size is
    %                  (channel_num + 1) * sample_size
    %        all_label: The labels that correspond to each sample. Its 
    %                   size is 1 * sample_size
    %        all_result_matrix: The matrix that uses one-hot encoding to
    %                           encode labels. Its size is
    %                           num_possible_labels * sample_size
    %        n_fold: an integer which is the number folds of cross 
    %                validation that will be performed
    % Output: test_accuracy: an array that stores the cross validation
    %                        testing accuracy result for each fold.
    %         train_accuracy: an array that stores the cross validation
    %                         training accuracy result for each fold.
    samples_per_iteration = 200;
    training_iterations = 1000;
    layer_sizes = [95,40,20,8];
    % layer_sizes = [95,20,8];
    
    layer_number = length(layer_sizes)-1;
    
    test_accuracy = [];
    train_accuracy = [];
    samples_per_fold = size(all_data, 2)  / n_fold;
    start_loc = 1;
    for i = 1:n_fold
        % Initialize a new instance of the NeuroNetClassifier class
        net = NeuroNetClassifier(layer_number, layer_sizes);
        
        % From all data, generate the n-fold cross validation data
        % The data are divided into training part and testing part
        int_start = round(start_loc);
        int_end = round(start_loc + samples_per_fold);
        if int_end > size(all_data, 2)
            int_end = size(all_data, 2);
        end
        test_length = int_end - int_start + 1;
        testing_data = all_data(:, int_start:int_end);
        testing_label = all_label(:, int_start:int_end);
        training_data = zeros(...
            size(all_data, 1), size(all_data, 2) - test_length);
        training_label = zeros(1, size(all_data, 2) - test_length);
        training_result_matrix = zeros(...
            size(all_result_matrix, 1), size(all_data, 2) - test_length);
        training_data(:, 1:int_start - 1) = all_data(:, 1:int_start - 1);
        training_label(:, 1:int_start - 1) = all_label(:, 1:int_start - 1);
        training_result_matrix(:, 1:int_start - 1) = ...
            all_result_matrix(:, 1:int_start - 1);
        if int_end ~= size(all_data, 2)
            training_label(:, int_start:end) = all_label(:,...
                                                         int_end + 1:end);
            training_data(:, int_start:end) = all_data(:, int_end + 1:end);
            training_result_matrix(:, int_start:end) =...
                all_result_matrix(:, int_end + 1:end);
        end
        % The index where the next fold's testing data will start
        start_loc = start_loc + samples_per_fold + 1;
        
        % The below line's code is used to set the training data set to be 
        % all the samples
%         samples_per_iteration = size(training_data, 2);

        %error = [];
        %train_accu = [];
        %test_accu = [];
        
        for iter = 1:training_iterations
            % For each iteration, draw random samples from the training
            % dataset, and use this set to train the model. This method
            % converges faster and reduced the training time alot.
            random_index = randi([1 size(training_data, 2)],...
                                 1, samples_per_iteration);
            iteration_training_data = training_data(:, random_index);
            iteration_training_label = training_result_matrix(...
                :,random_index);
            [new_coeff, new_sum_error] = net.BackPropagation(...
                iteration_training_data, iteration_training_label, 6);
            % The next two lines updates the class instance's properties
            % I didn't find a way to update properties from its method,
            % thus I added the two line here.
            net.coefficients = new_coeff;
            net.sum_gradient_square = new_sum_error;
        end
        % Calculate and store the training and testing accuracy
        output = ForwardPropagation(net, testing_data);
        [~, result_label] = max(output{layer_number}, [], 1);
        test_accuracy = [...
            test_accuracy...
            size(find(result_label - testing_label == 0), 2) /...
            size(testing_data, 2)];
        output = ForwardPropagation(net, training_data);
        [~, result_label] = max(output{layer_number}, [], 1);
        train_accuracy = [...
            train_accuracy...
            size(find(result_label - training_label == 0), 2) /...
            size(training_data, 2)];
        clear testing_data;
        clear testing_label;
        clear testing_result_matrix;
    end
    
    % Print performance metrics
    for fold = 1:n_fold
        fprintf(1,'Iteration #%0.0f:\n',fold);
        fprintf(1,'\tTraining accuracy:\t %0.3f\n',train_accuracy(fold));
        fprintf(1,'\tTesting accuracy:\t %0.3f\n',test_accuracy(fold));
    end