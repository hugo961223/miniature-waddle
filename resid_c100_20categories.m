%% residual network (baseline with skip layer)
netWidth = 16;
layers = [
    imageInputLayer([32 32 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    reluLayer('Name','relu12')
    
    convolutionalUnit(2*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    reluLayer('Name','relu22')
    
    convolutionalUnit(4*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    reluLayer('Name','relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(20,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'reluInp','add11/in2');
lgraph = connectLayers(lgraph,'relu11','add12/in2');
skip1 = [
    convolution2dLayer(1,2*netWidth,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,'relu12','skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');
lgraph = connectLayers(lgraph,'relu21','add22/in2');
skip2 = [
    convolution2dLayer(1,4*netWidth,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'relu22','skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');
lgraph = connectLayers(lgraph,'relu31','add32/in2');

% look at the structure
% figure('Units', 'normalized', 'Position',[0.2 0.2 0.6 0.6]);
% plot(lgraph);

%% train network
miniBatchSize = 128;
learnRate = 0.01;
valFrequency = floor(size(c100_xtrain_20,4)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', miniBatchSize, ...
    'VerboseFrequency', valFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', {c100_xtest_20, c100_ytest_20}, ...
    'ValidationFrequency', valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 60);

trainedNet_resid_c100_20 = trainNetwork(augimdsTrain_c100_20, lgraph, options);

%% evaluation
[YValPred,probs] = classify(trainedNet_resid_c100_20,c100_xtest_20);
validationError = mean(YValPred ~= c100_ytest_20);
YTrainPred = classify(trainedNet_resid_c100_20,c100_xtrain_20);
trainError = mean(YTrainPred ~= c100_ytrain_20);
% training error
disp("Training error: " + trainError*100 + "%")
% testing error
disp("Validation error: " + validationError*100 + "%")
% Confusion Matrix
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(c100_ytest_20,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
% more straightforward
figure
idx = randperm(size(c100_xtest_20,4),9);
for i = 1:numel(idx)
    subplot(3,3,i)
    imshow(c100_xtest_20(:,:,:,idx(i)));
    prob = num2str(100*max(probs(idx(i),:)),3);
    predClass = char(YValPred(idx(i)));
    title([predClass,', ',prob,'%'])
end