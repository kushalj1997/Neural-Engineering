% BIOMEDE 517 - Neural Engineering
% Lab 5 Part 3
% Kushal Jaligama

% Part 3
% Dimensionality Reduction of Spike Recordings
close all
% Step 1, normalize the data to have mean of 0 and standard deviation of 1
normalized_spikes = spikes;
for i=1:size(spikes,2)
    normalized_spikes(:,i) = (spikes(:,i)-mean(spikes(:,i))) / std(spikes(:,i));
end

figure(5)
plot(linspace(1, 124, 41568), spikes)

% Time axis for the reconstructed spike snippets
time_axis = linspace(0, 3, 32); % Each snippet is 3 ms long and has 32 samples

% Get the PCA components, u is eigenvectors (these represent the principal components of the dataset)
% w is scores, latent is eigenvals
[u, w, eigenvals] = pca(normalized_spikes);

% Determine how many of the princip components capture 90% of variance
covariance = cumsum(eigenvals)/sum(eigenvals);
% The first covariance component that has .9 is covariance(9)
K = 9;
% Pick a representative spike and plot the top k eigenvectors of the data
% These eigenvectors correspond to the k largest eigenvalues
spike_num = 5; % We are asked to analyze spike number 5
% Perform a matrix multiplication of the scores and the first 9 PC vectors
spike_one = w(1, 1:K)*transpose(u(:,1:K)); % Grab 9 columns of eigenvectors and transpose
figure(1);
plot(0:1:31, spike_one);

% Plot reconstructed spikes based on different numbers of prinicpal
% components

subplot_x = 4; % How many rows there are
subplot_y = 1; % How many columns there are
subplot_num = 1;

num_pcas = [9 32 6 4];
figure(2);
for figs=1:4
    K = num_pcas(figs);
    reconstructed_spike = w(spike_num, 1:K) * transpose(u(:, 1:K));
    subplot(subplot_x, subplot_y, subplot_num);
    title(sprintf('Reconstruction of Spike Using %f Principal Components', num_pcas));
    plot(reconstructed_spike);
    subplot_num = subplot_num + 1;
end

% Extract first two principal components for all the data
first_two_princips = w(:, 1:2);

figure(3)
scatter(w(:, 1), w(:, 2));
title('Comparing First 2 Principal Components of Each Data Point')

