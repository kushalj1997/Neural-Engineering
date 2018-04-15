% BIOMEDE 517 - Neural Engineering
% Lab 7 Part 2 - Linear Discriminant Analysis
% Kushal Jaligama

clearvars
close all

% Load data from ecogclassifydata.mat
load('ecogclassifydata.mat')
% Average power values from 0.5s prior to movement to 1.5s after
% movement in 60-120 Hz band
numTrials = 39;
% 27 electrode pairs, 'group' indicates which finger
% 1 is rest, 2 is thumb, 3 is index, 4 is middle
% 5 is ring and pinkie.

% Make LDA prediction of test sample's class
% Iterate through data and give each sample a turn at being test data
% Rest of samples should be training
for i = 1:numTrials
    test = powervals(i, :);
    training = powervals(1:i, :);
    training(end:numTrials - 1, :) = powervals(i + 1:end, :);
    train_groups = group(1:i, :);
    train_groups(end:numTrials - 1, :) = group(i + 1:end, :);
    predictions(i) = classify(test, training, train_groups, 'linear');
end

predictions = transpose(predictions);
num_correct = 0;

for i = 1:numTrials
    if predictions(i) == group(i)
        num_correct = num_correct + 1;
    end
end

accuracy = num_correct / numTrials

% This tells you what points were classified correctly and not
conf = confusionmat(group, predictions);
% A value of 1 at (2,1) in conf means one value that was supposed
% to be in class 2 was misclassified to group 1

% Plot this for visualization
imagesc(conf)

% Calculate percent correct by class
for i = 1:5
    total_in_class = sum(conf(i,:));
    percent_correct(i) = conf(i, i) / total_in_class;
end

percent_correct
