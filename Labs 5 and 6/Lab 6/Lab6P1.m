% BIOMEDE 517 - Neural Engineering
% Lab 6 Part 1
% Kushal Jaligama

% Run Lab5P2.m first so that the PCA is done on data and variables are
% stored into the workspace memory

% Pull out the first two principal components of the spike waveform data
% Plot every 4th point for more clarity in data separation
scatter(w(1:4:41568, 1), w(1:4:41568, 2));
figure (2)
scatter(w(:, 1), w(:, 2));

% Looking at the graphs it seems like there are two clusters close to each
% other with a third cluster lingering

K = 3;



