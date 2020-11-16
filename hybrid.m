

imds = imageDatastore(fullfile('/Users/mmabubakar/documents/matlab/xrayscan/training'),...
    'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.png','.jpeg'},...
    'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');
numTrainImages = numel(imdsTrain.Labels);

%net = alexnet;
net = resnet18;

inputSize = net.Layers(1).InputSize;
netlayer = 2;
name = net.Layers(netlayer).Name;
analyzeNetwork(net);

augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);

%augimdsTrain = augmentedImageDatastore(inputSize(1:3),imdsTrain );
%augimdsTest = augmentedImageDatastore(inputSize(1:3),imdsTest);

augimdsTrain = augmentedImageDatastore([224 224],...
    imdsTrain,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');

augimdsTest = augmentedImageDatastore([224 224],...
    imdsTest,'DataAugmentation',augmenter,'ColorPreprocessing','gray2rgb');

layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
whos featuresTrain
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
%classifier = fitcknn(featuresTrain,YTrain,'NumNeighbors',3);
%classifier = fitcnb(featuresTrain,YTrain);
%classifier = fitcecoc(featuresTrain,YTrain);
%classifier = fitcecoc(featuresTrain,YTrain,'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%    'expected-improvement-plus'))

classifier = fitctree(featuresTrain,YTrain);
%classifier = fitclinear(featuresTrain,YTrain);

L = loss(classifier,featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest)
%figure
%confusionchart (YTest, YPred, 'Normalization','total-normalized')


figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart (YTest, YPred, 'Normalization','total-normalized');
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm1 = confusionchart (YTest, YPred);
cm1.Title = 'Confusion Matrix for Test Data';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';
%figure;
%hold on
%plot(tr.TrainingAccuracy, '-', 'LineWidth', 1)
%plot(tr.ValidationAccuracy, 'o', 'LineWidth', 1)
%hold off
%xlabel('Iteration')
%ylabel('Accuracy')

error = resubLoss(classifier)

CVModel = crossval(classifier)
