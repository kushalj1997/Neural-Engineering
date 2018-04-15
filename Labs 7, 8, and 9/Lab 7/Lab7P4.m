% BIOMEDE 517 - Neural Engineering
% Lab 7 Part 4 - Multi-Class Support-Vector Machine
% Kushal Jaligama

clearvars
close all

% Load ECoG data
load('ecogclassifydata.mat');

numTrials = 39;
numGroups = 5;

% Create a multi-class SVM model that can use int v
SVMmodel = fitcecoc(powervals, group, 'Leaveout', 'on', 'Coding', 'onevsall');
predictions = kfoldPredict(SVMmodel);

num_correct = 0;
for i = 1:numTrials
    if predictions(i) == group(i)
        num_correct = num_correct + 1;
    end
end

disp('onevsall')
accuracy = num_correct / numTrials
conf1 = confusionmat(group, predictions);

% onevsone - force each SVM to be tested against all non-target
% classes separately in the model
SVMmodel = fitcecoc(powervals, group, 'Leaveout', 'on', 'Coding', 'onevsone');
predictions = kfoldPredict(SVMmodel);

num_correct = 0;
for i = 1:numTrials
    if predictions(i) == group(i)
        num_correct = num_correct + 1;
    end
end

disp('onevsone')
accuracy = num_correct / numTrials
conf2 = confusionmat(group, predictions);

% ternarycomplete - partitions n classes into positive, negative, 
% and zero valued classes that are cycled during training
SVMmodel = fitcecoc(powervals, group, 'Leaveout', 'on', 'Coding', 'ternarycomplete');
predictions = kfoldPredict(SVMmodel);

num_correct = 0;
for i = 1:numTrials
    if predictions(i) == group(i)
        num_correct = num_correct + 1;
    end
end

disp('ternarycomplete')
accuracy = num_correct / numTrials
conf3 = confusionmat(group, predictions);

% ordinal - uses n-1 binary SVMs for n classes. sampling space is
% partitioned at each class threshold
SVMmodel = fitcecoc(powervals, group, 'Leaveout', 'on', 'Coding', 'ordinal');
predictions = kfoldPredict(SVMmodel);

num_correct = 0;
for i = 1:numTrials
    if predictions(i) == group(i)
        num_correct = num_correct + 1;
    end
end

disp('ordinal')
accuracy = num_correct / numTrials
conf4 = confusionmat(group, predictions);

figure(1)
imagesc(conf1)
figure(2)
imagesc(conf2)
figure(3)
imagesc(conf3)
figure(4)
imagesc(conf4)
