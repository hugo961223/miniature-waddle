%% deep network
netWidth = 16;
layers = [
   imageInputLayer([32 32 3],'Name','input')
   convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
   reluLayer('Name','reluInp')
   
   convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','S1U1_conv1')
   reluLayer('Name','S1U1_relu1')
   convolution2dLayer(3,netWidth,'Padding','same','Name','S1U1_conv2')
   reluLayer('Name','relu11')
   convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','S1U2_conv1')
   reluLayer('Name','S1U2_relu1')
   convolution2dLayer(3,netWidth,'Padding','same','Name','S1U2_conv2')
   reluLayer('Name','relu12')
   
   convolution2dLayer(3,2*netWidth,'Padding','same','Stride',2,'Name','S2U1_conv1')
   reluLayer('Name','S2U1_relu1')
   convolution2dLayer(3,2*netWidth,'Padding','same','Name','S2U1_conv2')
   reluLayer('Name','relu21')
   convolution2dLayer(3,2*netWidth,'Padding','same','Stride',1,'Name','S2U2_conv1')
   reluLayer('Name','S2U2_relu1')
   convolution2dLayer(3,2*netWidth,'Padding','same','Name','S2U2_conv2')
   reluLayer('Name','relu22')
   
   convolution2dLayer(3,4*netWidth,'Padding','same','Stride',2,'Name','S3U1_conv1')
   reluLayer('Name','S3U1_relu1')
   convolution2dLayer(3,4*netWidth,'Padding','same','Name','S3U1_conv2')
   reluLayer('Name','relu31')
   convolution2dLayer(3,4*netWidth,'Padding','same','Stride',1,'Name','S3U2_conv1')
   reluLayer('Name','S3U2_relu1')
   convolution2dLayer(3,4*netWidth,'Padding','same','Name','S3U2_conv2')
   reluLayer('Name','relu32')

   averagePooling2dLayer(8,'Name','globalPool')
   fullyConnectedLayer(10,'Name','fcFinal')
   softmaxLayer('Name','softmax')
   classificationLayer('Name','classoutput')
   ];

% look at the structure
lgraph_deep = layerGraph(layers);
% figure('Units', 'normalized', 'Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph_deep);

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

trainedNet_deep = trainNetwork(augimdsTrain, lgraph_deep, options);

%% evaluation
[YValPred,probs] = classify(trainedNet_deep,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet_deep,XTrain);
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