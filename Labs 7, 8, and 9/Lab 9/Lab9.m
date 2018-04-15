% BIOMEDE 517 - Neural Engineering
% Lab 9 Part 1 - LASSO
% Kushal Jaligama

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


% LASSO

optimal_lambda = 0.0286

% Set up the B matrix
B = zeros(951, 4);
for i = 1:4
    % Get the weights for each recorded unit (neurons)
    B(:, i) = lasso(training_y, training_x(:,i), 'Lambda', optimal_lambda);
end

lasso_predictions = test_y * B;

% Now calculate the mean squared error and correlation coefficients
% Measure the mean squared error of predictions on the test data
lasso_mean_squared_error = sum(mean((lasso_predictions - test_x).^2))
% Correlation of predictions to actual motion
for i = 1:4
    lasso_corr_coeffs(i) = corr2(test_x(:, i), lasso_predictions(:, i));
end
lasso_corr_coeffs
