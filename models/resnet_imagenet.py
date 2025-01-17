'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch.nn as nn
import math
import pdb
import torch



def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, layers[0])

        if len(layers) == 3:
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        elif len(layers) == 2:
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
            self.avgpool = nn.AvgPool2d(16, stride=1)
            self.fc = nn.Linear(32 * block.expansion, num_classes)

        else:
            # layers == 1
            self.avgpool = nn.AvgPool2d(32, stride=1)
            # self.fc = nn.Linear(16 * block.expansion, num_classes)
            self.fc = nn.Linear(16 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        print(f"input: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        if 'layer2' in self._modules:
            print("here?")
            x = self.layer2(x)

        if 'layer3' in self._modules:
            x = self.layer3(x)
        print(f"2: {x.shape}")


        x = self.avgpool(x)
        print(f"3: {x.shape}")

        x = x.view(x.size(0), -1)
        print(f"4: {x.shape}")

        x = self.fc(x)
        print(f"5: {x.shape}")

        return x


class PreAct_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# def resnet4(**kwargs):
#     model = ResNet(BasicBlock, [1], **kwargs)
#     return model


def resnet6_imagenet(**kwargs):
    model = ResNet(BasicBlock, [1, 1], **kwargs)
    return model

# def resnet8(**kwargs):
#     model = ResNet(BasicBlock, [1, 1, 1], **kwargs)
#     return model


# def resnet14(**kwargs):
#     model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
#     return model


# def resnet20(**kwargs):
#     model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
#     return model


# def resnet32(**kwargs):
#     model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
#     return model


# def resnet44(**kwargs):
#     model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
#     return model


# def resnet56(**kwargs):
#     model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
#     return model


# def resnet110(**kwargs):
#     model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
#     return model


# def resnet1202(**kwargs):
#     model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
#     return model


# def resnet164(**kwargs):
#     model = ResNet(Bottleneck, [18, 18, 18], **kwargs)
#     return model


# def resnet1001(**kwargs):
#     model = ResNet(Bottleneck, [111, 111, 111], **kwargs)
#     return model


# def preact_resnet110(**kwargs):
#     model = PreAct_ResNet(PreActBasicBlock, [18, 18, 18], **kwargs)
#     return model


# def preact_resnet164(**kwargs):
#     model = PreAct_ResNet(PreActBottleneck, [18, 18, 18], **kwargs)
#     return model


# def preact_resnet1001(**kwargs):
#     model = PreAct_ResNet(PreActBottleneck, [111, 111, 111], **kwargs)
#     return model


if __name__ == "__main__":
    net = resnet6_imagenet(num_classes=1000)
    # print(net)
    total_params = sum(p.numel() for p in net.parameters())
    layers = len(list(net.modules()))
    print(f" total parameters: {total_params}, layers {layers}")
    input = torch.rand([128, 3, 224, 224])
    net(input)
