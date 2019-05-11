%% shallow network with batchnorm
netWidth = 16;
layers = [
   imageInputLayer([32 32 3],'Name','input')
   convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
   batchNormalizationLayer('Name','BNInp')
   reluLayer('Name','reluInp')
   
   convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','S1_conv')
   batchNormalizationLayer('Name','S1_BN')
   reluLayer('Name','S1_relu')
   
   convolution2dLayer(3,2*netWidth,'Padding','same','Stride',2,'Name','S2_conv')
   batchNormalizationLayer('Name','S2_BN')
   reluLayer('Name','S2_relu')
   
   convolution2dLayer(3,4*netWidth,'Padding','same','Stride',2,'Name','S3_conv')
   batchNormalizationLayer('Name','S3_BN')
   reluLayer('Name','S3_relu')

   averagePooling2dLayer(8,'Name','globalPool')
   fullyConnectedLayer(10,'Name','fcFinal')
   softmaxLayer('Name','softmax')
   classificationLayer('Name','classoutput')
   ];

% look at the structure
lgraph_shallow_BN = layerGraph(layers);
figure('Units', 'normalized', 'Position',[0.2 0.2 0.6 0.6]);
plot(lgraph_shallow_BN);

%% train network
miniBatchSize = 128;
learnRate = 0.01;
valFrequency = floor(size(XTrain,4)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', miniBatchSize, ...
    'VerboseFrequency', valFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 60);

trainedNet_shallow_BN = trainNetwork(augimdsTrain, lgraph_shallow_BN, options);

%% evaluation
[YValPred,probs] = classify(trainedNet_shallow_BN,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet_shallow_BN,XTrain);
trainError = mean(YTrainPred ~= YTrain);
% training error
disp("Training error: " + trainError*100 + "%")
% testing error
disp("Validation error: " + validationError*100 + "%")
% Confusion Matrix
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
% more straightforward
figure
idx = randperm(size(XValidation,4),9);
for i = 1:numel(idx)
    subplot(3,3,i)
    imshow(XValidation(:,:,:,idx(i)));
    prob = num2str(100*max(probs(idx(i),:)),3);
    predClass = char(YValPred(idx(i)));
    title([predClass,', ',prob,'%'])
end