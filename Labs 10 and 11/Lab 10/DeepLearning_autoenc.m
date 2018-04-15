%%Runs a deep neural net with two stacked encoder
%layers followed by a softmax layer.

close all;
[all_data, all_label, all_result_matrix] = GetAllData('firingrate.mat');
%Reformat labels to fit neural network toolbox
final_labels=zeros(8,length(all_label));
for i=1:length(all_label)
    final_labels(all_label(1,i),i)=1;
end

%Run deep net at different percents and plot testing accuracy from
%confusion matrix

% trainvals = linspace(0.01, 0.9, 10); % Test 10 different train vals
trainvals = [0.01 0.3 0.6 0.9]
accvals = [94.9 92.1 95.4 92.9 98.2 94.1 97.8 95.9 95 94.1]
inaccvals = [5.1 7.9 4.6 7.1 1.8 5.9 2.2 4.1 5.0 5.9]
accvals2 = [17.6 97.1 97.9 98.6]
inaccvals2 = [82.4 2.9 2.1 1.4]
accvals3 = [12.3 12.2 11.5 8.9]
inaccvals3 = [87.7 87.8 88.5 91.1]

figure(1)
hold on
% Plot the autoencoder [20 10]
plot(trainvals, accvals2)
plot(trainvals, inaccvals2)

% Plot the autoencoder [25 20 15 10]
plot(trainvals, accvals3)
plot(trainvals, inaccvals3)

% for i = trainvals
    %split into test and train data
    perTrain=0.3; %switch how much of data to use for training
    N=size(all_data,2);
    randInds=randperm(N);
    trainData=all_data(:,randInds(1:floor(N*perTrain)));
    testData=all_data(:,randInds(floor(N*perTrain)+1:end));
    trainLabels=final_labels(:,randInds(1:floor(N*perTrain)));
    testLabels=final_labels(:,randInds(floor(N*perTrain)+1:end));

    %Train first autoencoder with 10 neurons
    hiddenSizes=[25 20 15 10];
    features1=trainData; autoencStr=''; numAutoencoders=length(hiddenSizes);
    for i=1:numAutoencoders
        scalDat=true;
        if i>1
            scalDat=false;
        end
        autoenc0 = trainAutoencoder(features1,hiddenSizes(i),...
            'L2WeightRegularization',0.001,...
            'SparsityRegularization',4,...
            'SparsityProportion',0.05,...
            'DecoderTransferFunction','purelin','ScaleData',scalDat);
        features1 = encode(autoenc0,features1);
        eval(['autoenc' num2str(i) '=autoenc0;']);
        autoencStr=[autoencStr 'autoenc' num2str(i) ','];
    end

    %Train softax layer using features from autoencoder and training labels
    softnet = trainSoftmaxLayer(features1,trainLabels,'LossFunction','crossentropy');

    %Create a deep neural net with the stacked layers
    eval(['deepnet = stack(' autoencStr 'softnet);']);

    %Train the neural net
    deepnet = train(deepnet,trainData,trainLabels);

    %Predict values from the test set
    predictedResults = deepnet(testData);
    % disp(predictedResults)

    %Plot the confusion matrix to see how well we did
    plotconfusion(testLabels,predictedResults);
    % size(predictedResults)
    % pause(5)

    % Plot autoencoder weights
    figure; 
    for i=1:numAutoencoders
        subplot(numAutoencoders,1,i);
        eval(['plotWeights(autoenc' num2str(i) ')']);
    end

% end