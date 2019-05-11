%% deeper residual network (baseline with skip layer and extra layer)
numUnits = 9;
netWidth = 16;
lgraph = residualCIFARlgraph(netWidth, numUnits, "standard");


% look at the structure
figure('Units', 'normalized', 'Position',[0.1 0.1 0.8 0.8]);
plot(lgraph);

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

trainedNet_resid_deep = trainNetwork(augimdsTrain, lgraph, options);

%% evaluation
[YValPred,probs] = classify(trainedNet_resid_deep,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet_resid_deep,XTrain);
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