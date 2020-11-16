%% LOAD PRE-TRAINED NETWORK
network = alexnet;
%network = resnet18;

%% LOADING DATA

imds = imageDatastore(fullfile('/Users/mmabubakar/documents/matlab/xrayscan/training'),...
    'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.png','.jpeg'},...
    'LabelSource','foldernames');

table = countEachLabel(imds);

imagenames = imds.Labels;

numClasses = numel(categories(imds.Labels));


%% LAYER MODIFICATION (Transfer Learning)

layers = network.Layers;

layers(23) = fullyConnectedLayer(numClasses); %AlexNet
%layers(69) = fullyConnectedLayer(numClasses); %ResNet

layers(25) = classificationLayer; %AlexNet
%layers(71) = classificationLayer; %ResNet


inputSize = network.Layers(1).InputSize;
analyzeNetwork(network);


%% PREPARING TRAINING & TEST DATA

% Data Augumentation
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);

[imdsTrain,imdsTest,imdsValidation] = splitEachLabel(imds,0.7,0.2,'randomized');

%,imdsValidation
%[audsTrain,audsTest] = splitEachLabel(auds,250,'randomized')

audsTrain = augmentedImageDatastore([227 227],...
imdsTrain,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');

audsTest = augmentedImageDatastore([227 227],...
imdsTest,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');

audsValidation = augmentedImageDatastore([227 227],...
imdsValidation,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');

cat = imdsValidation.Labels;

%% SETTING TRAINING OPTIONS

options = trainingOptions('sgdm','MaxEpochs',20,'InitialLearnRate',0.001, ...
    'ValidationData',{audsValidation,cat},'ValidationFrequency',5,'Plots','training-progress');

%'ValidationData',{audsValidation},'ValidationFrequency',50,'MaxEpochs',10


%% PERFORM TRAINING

[finalnet,info] = trainNetwork(audsTrain, layers, options);

montage(imds);

%% USE TRAINED NETWORK TO CLASSIFY TEST DATA

testpred = classify(finalnet,audsTest)

%% NETWORK PERFORMANCE STUDY
% Confusion Matrix
%title('Confusion Matrix: AlexNet');
