%% batchnorm network
netWidth = 16;
layers = [
    imageInputLayer([32 32 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    reluLayer('Name','relu12')
    
    convolutionalUnit(2*netWidth,2,'S2U1')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    reluLayer('Name','relu22')
    
    convolutionalUnit(4*netWidth,2,'S3U1')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    reluLayer('Name','relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(10,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

% look at the structure
lgraph_batchnorm = layerGraph(layers);
% figure('Units', 'normalized', 'Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph_batchnorm);

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

trainedNet_batchnorm = trainNetwork(augimdsTrain, lgraph_batchnorm, options);

%% evaluation
[YValPred,probs] = classify(trainedNet_batchnorm,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet_batchnorm,XTrain);
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