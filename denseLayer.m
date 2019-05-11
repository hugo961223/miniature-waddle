function layers = denseLayer(in_planes,num,tag)
layers = [];

for i = 1:num
    layers = [layers
        batchNormalizationLayer('Name',[tag,i,'BN1'])
        convolution2dLayer(1,48,'Name',[tag,i,'conv1'])
        batchNormalizationLayer('Name',[tag,i,'BN2'])
        convolution2dLayer(3,12,'Name',[tag,i,'conv2'])];
end
end