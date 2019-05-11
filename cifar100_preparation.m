%% load cifar-100 dataset
% Apple - 1(0)              | Caterpillar - 19(18)
% Chair - 21(20)            | Baby - 3(2)
% Keyboard - 40(39)         | Dinosaur - 30(29)
% Oak_tree - 53(52)         | Mouse - 51(50)
% Spider - 80(79)           | Bicycle - 9(8)
% Ray - 68(67)              | Rose - 71(70)
% House - 38(37)            | Kangaroo 39(38)
% Dolphin - 31(30)          | Cup - 29(28)
% Bear - 4(3)               | Sea - 72(71)
% Fox - 35(34)              | Tank - 86(85)

c100_train = load('cifar100_train.mat');
c100_test = load('cifar100_test.mat');
c100_meta = load('cifar100_meta.mat');

label = [0 20 39 52 79];
label_20 = [0 18 20 2 39 29 52 50 79 8 67 70 37 38 30 28 3 71 34 85];
idx_label = find(ismember(c100_train.fine_labels,label));
idx_label_20 = find(ismember(c100_train.fine_labels,label_20));
idx_label_test = find(ismember(c100_test.fine_labels,label));
idx_label_test_20 = find(ismember(c100_test.fine_labels,label_20));


[c100_xtrain,c100_ytrain] = loadCIFAR100Data(c100_train,c100_meta,idx_label,label);
[c100_xtrain_20,c100_ytrain_20] = loadCIFAR100Data(c100_train,c100_meta,idx_label_20,label_20);
[c100_xtest,c100_ytest] = loadCIFAR100Data(c100_test,c100_meta,idx_label_test,label);
[c100_xtest_20,c100_ytest_20] = loadCIFAR100Data(c100_test,c100_meta,idx_label_test_20,label_20);



% display images
figure;
idx = randperm(size(c100_xtrain_20,4), 20);
im = imtile(c100_xtrain_20(:,:,:,idx), 'ThumbnailSize', [96,96]);
imshow(im);

%% data augmentation for 5 categories

imageSize = [32 32 3];
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain_c100_5 = augmentedImageDatastore(imageSize,c100_xtrain,c100_ytrain, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');

%% data augmentation for 20 categories

imageSize = [32 32 3];
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain_c100_20 = augmentedImageDatastore(imageSize,c100_xtrain_20,c100_ytrain_20, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');

%% help function

function [X,Y] = loadCIFAR100Data(Data,meta,index,label)

XBatch = Data.data(index,:)';
X = reshape(XBatch,32,32,3,[]);
X = permute(X,[2 1 3 4]);
Y = categorical(Data.fine_labels(index),label,meta.fine_label_names(label+1));

end