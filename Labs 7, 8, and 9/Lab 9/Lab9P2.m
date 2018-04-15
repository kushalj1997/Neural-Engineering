% BIOMEDE 517 - Neural Engineering
% Lab 9 All Parts - LASSO and Kalman Filters
% Kushal Jaligama

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

% Kalman Filter

% The physics is represented as:
% x_t is a state ("position, velocity")
% x_t = Ax_(t-1) + w_t
%       physics    noise

% y_t is neural firing rates at time step
% y_t = C_x(t) + q_t
%       LinFilt  noise

% Transpose the matrices to put time domain as columns
training_x = transpose(training_x);
training_y = transpose(training_y);
test_x = transpose(test_x);
test_y = transpose(test_y);

% Set up the time step variables
training_x_prev = training_x;
% Remove first element of training data (ones)
training_x(:, 1) = [];
training_y(:, 1) = [];
% Remove last element of "prev" matrix for prediction purposes
training_x_prev(:, training_rows) = [];

% Get the physics and linear filter matrices
C = (training_y * transpose(training_x)) / (training_x * transpose(training_x));
A = (training_x * transpose(training_x_prev)) / (training_x_prev* transpose(training_x_prev));

% Get the noise
W = (1 / (numObservations - 1)) * (training_x - A * training_x_prev) * transpose(training_x - A * training_x_prev);
Q = (1 / (numObservations - 1)) * (training_y - C * training_x) * transpose(training_y - C * training_x);

% Start making predictions
xhat = zeros(4, test_rows);
xhat(:, 1) = test_x(:, 1);

% a posterior covariance of x, how accuracte is the estimate
postcov = W;

% Predict, innovate, update
for i = 2:test_rows
    % Prediction given last time step
    xhatprev = A * test_x(:, i - 1);
    postcovprev = A * postcov * transpose(A) + W;
    Kt = postcovprev * transpose(C) / (C * postcovprev * transpose(C) + Q);
    xhat(:, i) = xhatprev + Kt * (test_y(:, i) - C * xhat(:, i-1));
    postcov = (eye(4) - Kt * C) * postcovprev;
end

% Now calculate the accuracy of this entire thing
kalman_mean_squared_error = sum(mean(test_x - xhat).^2)
% Correlation of predictions to actual motion
for i = 1:4
    kalman_corr_coeffs(i) = corr2(test_x(i, :), xhat(i, :));
end
kalman_corr_coeffs
