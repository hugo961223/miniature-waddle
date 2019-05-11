%% Load the trained residual network

transfer_net = load('trained_resid.mat');
transfer_net = transfer_net.trainedNet_resid;

layers = transfer_net.Layers(1:50);

% 48th layer is fcFinal
% 49th layer is softmax
% 50th layer is classoutput
layers(48) = fullyConnectedLayer(20,'Name','fcFinal');
layers(49) = softmaxLayer('Name','softmax');
layers(50) = classificationLayer('Name','classoutput');

skip1 = transfer_net.Layers(51:52);
skip2 = transfer_net.Layers(53:54);

lgraph_transfer = layerGraph(layers);
lgraph_transfer = connectLayers(lgraph_transfer,'reluInp','add11/in2');
lgraph_transfer = connectLayers(lgraph_transfer,'relu11','add12/in2');

lgraph_transfer = addLayers(lgraph_transfer,skip1);
lgraph_transfer = connectLayers(lgraph_transfer,'relu12','skipConv1');
lgraph_transfer = connectLayers(lgraph_transfer,'skipBN1','add21/in2');
lgraph_transfer = connectLayers(lgraph_transfer,'relu21','add22/in2');

lgraph_transfer = addLayers(lgraph_transfer,skip2);
lgraph_transfer = connectLayers(lgraph_transfer,'relu22','skipConv2');
lgraph_transfer = connectLayers(lgraph_transfer,'skipBN2','add31/in2');
lgraph_transfer = connectLayers(lgraph_transfer,'relu31','add32/in2');

% look at the structure
figure('Units', 'normalized', 'Position',[0.2 0.2 0.6 0.6]);
plot(lgraph_transfer);

%% train network
miniBatchSize = 128;
learnRate = 0.01;
valFrequency = floor(size(c100_xtrain_20,4)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', miniBatchSize, ...
    'VerboseFrequency', valFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', {c100_xtest_20, c100_ytest_20}, ...
    'ValidationFrequency', valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20);

transferNet_c100_20 = trainNetwork(augimdsTrain_c100_20, lgraph_transfer, options);


%% evaluation
[YValPred,probs] = classify(transferNet_c100_20,c100_xtest_20);
validationError = mean(YValPred ~= c100_ytest_20);
YTrainPred = classify(transferNet_c100_20,c100_xtrain_20);
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