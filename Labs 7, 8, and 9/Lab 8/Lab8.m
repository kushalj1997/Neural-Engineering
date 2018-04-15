% BIOMEDE 517 - Neural Engineering
% Lab 8 All Parts - Continuous Decoders
% Kushal Jaligama

% Real-time decoding algorithms starting with linear regression
% Using data from a reach task

clearvars
close all

% Part 0 - Process Data
load('contdata.mat')
% Columns of X contain X position, Y position, X velocity, Y velocity
% Columns of Y contain firing rates of 950 recorded units
numObservations = 31413; % Number of time points

% Split data into training and test 50/50
training_rows = floor(numObservations / 2);
training_x = X(1:training_rows, :); % Position X, Y, Velocity X, Y
training_y = Y(1:training_rows, :); % Firing Rates of 950 units

test_rows = ceil(numObservations / 2);
test_x = X(test_rows:end, :);
test_y = Y(test_rows:end, :);

% Add a column of ones to the neural data to calculate an intercept
% term in the regression models
training_y = [ones(training_rows, 1) training_y];
test_y = [ones(test_rows, 1) test_y];

% Part 1 - Linear Regression

% Solve a linear regression equation with training data
% x = yB => B is the linear decoder matrix we want to find
training_y_t = transpose(training_y);
B = inv(training_y_t * training_y) * (training_y_t * training_x);

% To predict new data, multiply it by your linear decoder matrix, B
linear_predictions = test_y * B;

% Measure the mean squared error of predictions on the test data
linear_mean_squared_error = sum(mean((linear_predictions - test_x).^2))
% Correlation of predictions to actual motion
for i = 1:4
    linear_corr_coeffs(i) = corr2(test_x(:, i), linear_predictions(:, i));
end
linear_corr_coeffs

% Part 2 - Ridge Regression
least_error = intmax;
lambda_vals = linspace(0, 0.1, 15);
ridge_errors = zeros(1, 15);
% Perform ridge regression on all of the lambda values
for i = 1:15
    lambda = lambda_vals(i);
    square = training_y_t * training_y;
    B_ridge = inv(square + training_rows * lambda * eye(size(square))) * training_y_t * training_x;
    prediction_ridge = test_y * B_ridge;
    % Calculate the mean squared errors of predictions on test data
    ridge_errors(i) = sum(mean((test_x - prediction_ridge).^2));
    % Use error terms to find the best lambda value
    if ridge_errors(i) <= least_error
        optimal_lambda = lambda;
        least_error = ridge_errors(i);
    end
end

optimal_lambda
least_error

plot(lambda_vals, ridge_errors)
hold on
plot(optimal_lambda, least_error, 'ro')

% Use the best lambda value to get best prediction
square = (training_y_t * training_y);
B_ridge = inv(square + training_rows * optimal_lambda * eye(size(square))) * training_y_t * training_x;
best_ridge_predictions = test_y * B_ridge;
best_ridge_mean_squared_error = sum(mean((test_x - best_ridge_predictions).^2))
% Correlation of predictions to actual motion
for i = 1:4
    ridge_corr_coeffs(i) = corr2(test_x(:, i), best_ridge_predictions(:, i));
end
ridge_corr_coeffs

