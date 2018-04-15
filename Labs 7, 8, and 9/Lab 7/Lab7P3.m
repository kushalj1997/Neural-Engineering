% BIOMEDE 517 - Neural Engineering
% Lab 7 Part 3 - Support-Vector Machine
% Kushal Jaligama

clearvars
close all

% Load ECoG data
load('ecogclassifydata.mat');

numTrials = 39;
numGroups = 5;

% Restructure group to establish binary labels
Y = zeros(numTrials, numGroups);
for i = 1:numTrials
    for j = 1:numGroups
        if group(i) == j
            Y(i, j) = 1;
        end
    end
end

% Make predictions on the model with an SVM using leave-one-out
% cross validation
for i = 1:numGroups
    y = Y(:, i);
    SVMmodel = fitcsvm(powervals, y, 'KernelFunction', 'linear', 'Leaveout', 'on');
    predictions(:, i) = kfoldPredict(SVMmodel);
end

% Establish total number of correct predictions for accuracy testing
num_correct = 0;
% Establish number of correct preds in each class
by_class = zeros(1, 5);
for j = 1:numGroups
    for i = 1:numTrials
        if predictions(i, j) == Y(i, j)
            num_correct = num_correct + 1;
            by_class(j) = by_class(j) + 1;
        end
    end
    % Calculate the accuracy for each class
    by_class_accuracy(j) = by_class(j) ./ numTrials;
end

overall_accuracy = num_correct / (numTrials * numGroups)

by_class_accuracy

% Non-linear SVM with radial basis kernel function
for i = 1:numGroups
    y = Y(:, i);
    SVMmodel = fitcsvm(powervals, y, 'KernelFunction', 'rbf', 'Leaveout', 'on');
    nonlin_predic(:,i) = kfoldPredict(SVMmodel);
end

% Calculate overall prediction accuracy and by class
nonlin_num_correct = 0;
nonlin_by_class = zeros(1, 5);
for i = 1:numGroups
    for j = 1:numTrials
        if nonlin_predic(j, i) == Y(j, i)
            nonlin_num_correct = nonlin_num_correct + 1;
            nonlin_by_class(i) = nonlin_by_class(i) + 1;
        end
    end
    nonlin_by_class_accuracy(i) = nonlin_by_class(i) ./ numTrials;
end

nonlin_overall_accuracy = nonlin_num_correct / (numTrials * numGroups)

nonlin_by_class_accuracy
