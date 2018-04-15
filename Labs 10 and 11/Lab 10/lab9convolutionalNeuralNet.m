%% Lab 9: Transfer learning using a pre-defined convolutional neural network
clear;
%% Load Pretrained Network
%Load the pretrained network: LettersClassificationNet.mat, which was 
%trained on 28-by-28 grayscale letter images. The network classifies images
%into 'A', 'B', and 'C'.
load(fullfile(matlabroot,'examples','nnet','LettersClassificationNet.mat'))
%%
% Examine the network architecture (layers of neural net)
net.Layers

%% Load Training Data
% Load the digits sample data as an ImageDatastore object. The
% ImageDatastore object lets you store large image data, including data
% that does not fit in memory. The object also lets you efficiently read
% batches of images during training.
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% The data store contains 10000 images of digits 0-9 (28-by-28 pixels).

% Split the data set into training and test data sets
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.5,'randomize'); %0.1 == 10% train data

% Display 20 sample training digits.
numImages = numel(trainDigitData.Files);
idx = randperm(numImages,20); %randomly select index
for i = 1:20
    subplot(4,5,i)
    
    I = readimage(trainDigitData, idx(i));
    
    imshow(I)
end
numClasses = numel(categories(trainDigitData.Labels)); %number of different numbers
%% Transfer Layers to Target Network
%%%%%%%%%%%%%%%%%%%%%%%%%%STUDENT WORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract all the layers except the last three from the pretrained network.

% Add in new layers to be trained
    
%%%%%%%%%%%%%%%%%%%%%%%%%%STUDENT WORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Note that you would need to resize the images if they didn't match between
%the net and your data (in this case, they match).
%%
% Create the training options. For transfer learning, you want to keep the
% features from the early layers of the pretrained network (the transferred
% layer weights). Set 'InitialLearnRate' to a low value to slows down 
% learning on the transferred layers. In the previous step, the learn rate 
% for the fully connected layer is high to speed up learning on the new 
% final layers. This combination results in fast learning only on the new 
% layers while keeping the other layers fixed. Because we don't need to 
% many epochs, use small value for 'MaxEpochs'.
optionsTransfer = trainingOptions('sgdm', ...
    'MaxEpochs',5, ...
    'InitialLearnRate',0.0001);
%% Train the new network
%%%%%%%%%%%%%%%%%%%%%%%%%%STUDENT WORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use trainNetwork()
%%%%%%%%%%%%%%%%%%%%%%%%%%STUDENT WORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate the classification accuracy: the proportion of correctly
% classified instances in the test data
%%%%%%%%%%%%%%%%%%%%%%%%%%STUDENT WORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use classify()
%%%%%%%%%%%%%%%%%%%%%%%%%%STUDENT WORK%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Display sample test images with their predicted labels
idx = floor(linspace(5,length(YTest),9)); %501:500:5000;
figure
for i = 1:numel(idx)
    subplot(3,3,i)
    
    I = readimage(testDigitData, idx(i));
    label = char(YTest(idx(i)));
    
    imshow(I)
    title(label)
end

%% Deepdream analysis of fully connected layer
figure('name','Deepdream results');
for i=1:10
    layerNum=5; %looking at last fully-connected layer
    levels=4; iterations = 100;
    channels=i; %which class (aka. number) to look at
    %Note: substitute netTransfer with the name of your trained network
    I = deepDreamImage(netTransfer,layerNum,channels, ...
        'Verbose',false, ...
        'NumIterations',iterations,...
        'PyramidLevels',levels);
    subplot(4,3,i);
    imshow(I)
    title(netTransfer.Layers(end).ClassNames{i});
end