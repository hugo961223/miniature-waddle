function layers = transLayer(in_planes,out_planes,tag)

layers = [
    batchNormalizationLayer('Name',[tag,'BN1'])
    convolution2dLayer(1,out_planes,'Name',[tag,'conv1'])];

end

