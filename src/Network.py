
from retrain_modules import *
from Parameters import NNParam
#from Parameters import Network


HWA = False
class Net(nn.Module):
    '''
    VGG model
    '''
    def __init__(self,Dir="Parameters"):
        super(Net, self).__init__()
        self.Dir = Dir
        # default: Vgg11
        Network = NNParam()
        layers = len(Network)
        Features = []
        Classifier = []
        layer = 1
        for l in range(layers):


            # TODO: Network file to be modified, the list in it will change from 3D to 2D.
            if str(Network[l][0][0]) == 'Conv':
                conv = Conv2d_Q(in_channels=Network[l][0][1],out_channels=Network[l][0][2],kernel_size=Network[l][0][5],\
                                      padding=Network[l][0][8],stride=Network[l][0][7],layer=layer,Dir=self.Dir)
                layer += 1
                Features.append(conv)
            if str(Network[l][0][0]) == 'Maxpool':
                maxpool = Maxpool_S(kernel_size=Network[l][0][5],stride=Network[l][0][7],layer=layer,Dir=self.Dir)
                Features.append(maxpool)
            if str(Network[l][0][0]) == 'Avgpool':
                avgpool = Avgpool_S(kernel_size=Network[l][0][5],stride=Network[l][0][7],layer=layer,Dir=self.Dir)
                Features.append(avgpool)
            if str(Network[l][0][0]) == 'ConvRelu':
                relu = Relu_S(inplace=True,layer=layer,Dir=self.Dir)
                Features.append(relu)
            if str(Network[l][0][0]) == 'ConvSigmoid':
                sigmoid = nn.Sigmoid(inplace=True,layer=layer)
                Features.append(sigmoid)
            if str(Network[l][0][0]) == 'Linear':
                fc = Linear_Q(in_features=Network[l][0][1],out_features=Network[l][0][2],layer=layer,Dir=self.Dir)
                layer += 1
                Classifier.append(fc)
            if str(Network[l][0][0]) == 'LinearRelu':
                relu = Relu_S(inplace=True,layer=layer,Dir=self.Dir)
                Classifier.append(relu)

        self.features = nn.Sequential(*Features)
        self.classifier = nn.Sequential(*Classifier)
        for m in self.modules():
            if isinstance(m, Conv2d_Q):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


