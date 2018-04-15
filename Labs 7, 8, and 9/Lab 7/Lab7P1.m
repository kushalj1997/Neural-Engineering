% BIOMEDE 517 - Neural Engineering
% Lab 7 Part 1 - Naïve Bayes Classifier with Poisson Distribution
% Kushal Jaligama

% Predicting reach direction of monkey using 95 neurons

clearvars
close all

% Load data from firingrate.mat
% 95 neurons, 8 reach dirs, 182 samples for each neuron-dir comobo
load('firingrate.mat')

training_samples = 91;
total_samples = 182;
test_samples = total_samples - training_samples;
% Split data in half, training and testing test data sets
% Since we have 182 samples, use training_samples samples for each half
training_data = firingrate(:, 1:training_samples, :);
test_data = firingrate(:, training_samples+1:total_samples, :);

% Training Step - Parameters for Poisson Distribution
% Calculate lambda indexed by neuron, target
lambda = zeros(95, 8); % n x m matrix
% mean firing rate of neuron n when reaching to direction m
for dir = 1:8
    for j = 1:95
        lambda(j, dir) = mean(training_data(j, :, dir));
    end
end

% Prediction Step - Testing the Classifier
% Loop through all individual trials of test data
predictions = zeros(test_samples, 8); % This is to assign the features

for class = 1:8
    for i = 1:test_samples
        % Get the feature vector (95 neurons, i'th sample, class)
        X_i = test_data(:, i, class);
        for k = 1:8
            % Calculate log prob density for the feature vector
            g_hat = poisspdf(X_i, lambda(:,k));
            % We will assign sample based on sum of probabilities
            arg_k(k) = sum(log(g_hat));
        end
        % Assign feature to maximal k class
        % (best direction for this set of neuron firing rates)
        [maximal_k, index] = max(arg_k);
        predictions(i, class) = index;
    end
end

% Calculate accuracy of class assignments
number_correct = 0;
number_total = 0;
for class = 1:8
    for i = 1:test_samples
        if predictions(i, class) == class
            number_correct = number_correct + 1;
        end
        number_total = number_total + 1;
    end
end
accuracy = number_correct / number_total

% Applying classifier to spoofed dataset
% Calculate lambda indexed by neuron, target for full dataset
lambda_full = zeros(95, 8); % n x m matrix
for dir = 1:8
    for j = 1:95
        lambda_full(j, dir) = mean(firingrate(j, :, dir));
    end
end

% Generate random data along a Poisson Distribution
spoof_data = zeros(95, total_samples, 8);
for i = 1:total_samples
    spoof_data(:,i,:) = poissrnd(lambda_full);
end

% Prediction Step - Testing the Classifier
% Loop through all individual trials of test data
predictions_full = zeros(total_samples, 8); % This is to assign the features

for class = 1:8
    for i = 1:total_samples % samples
        % Get the feature vector
        X_i_LDA = spoof_data(:, i, class);
        for k = 1:8
            % Calculate log prob density for the feature vector
            g_hat_LDA = poisspdf(X_i_LDA, lambda_full(:,k));
            % We will assign sample based on sum of probabilities
            arg_k_LDA(k) = sum(log(g_hat_LDA));
        end
        % Assign feature to maximal k class
        % (best direction for this set of neuron firing rates)
        [maximal_k, index] = max(arg_k_LDA);
        predictions_full(i, class) = index;
    end
end

% Calculate accuracy of class assignments
number_correct = 0;
number_total = 0;
for class = 1:8
    for i = 1:total_samples
        if predictions_full(i, class) == class
            number_correct = number_correct + 1;
        end
        number_total = number_total + 1;
    end
end
accuracy_full = number_correct / number_total
