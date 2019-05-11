%% data import
[XTrain, YTrain, XValidation, YValidation] = loadCIFARData('home/hhc/Downloads/');

%% display images
figure;
idx = randperm(size(XTrain,4), 20);
im = imtile(XTrain(:,:,:,idx), 'ThumbnailSize', [96,96]);
imshow(im);

%% data augmentation
imageSize = [32 32 3];
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');