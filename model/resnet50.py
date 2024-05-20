import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
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

class ResNet50(nn.Module):
    def __init__(self,block,layers,num_particle,num_class):
        self.inplanes = 64
        self.num_class = num_class
        self.num_particle = num_particle

        super(ResNet50,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.layer3s = nn.ModuleList([self._make_layer(block, 256, layers[2], stride=2) for _ in range(num_particle)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, 512, layers[3], stride=2) for _ in range(num_particle)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.linears = nn.ModuleList([NormedLinear(512*4, num_class) for _ in range(num_particle)])

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride!=1 or self.inplanes!=planes*4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*4),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes*4
        for i in range(1,blocks):
            layers.append(block(self.next_inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        self.logits = []

        for i in range(self.num_particle):
            xi = self.layer3s[i](x)
            xi = self.layer4s[i](xi)
            xi = self.avgpool(xi)
            xi = xi.view(xi.size(0),-1)
            xi = self.linears[i](xi)
            self.logits.append(xi*30)

        return sum(self.logits)/len(self.logits)

class ResNet50slim(nn.Module):
    def __init__(self,block,layers,num_particle,num_class):
        self.inplanes = 64
        self.num_class = num_class
        self.num_particle = num_particle

        super(ResNet50slim,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes
        self.layer3 = self._make_layer(block, 192, layers[2], stride=2)
        self.inplanes = self.next_inplanes

        self.layer4s = nn.ModuleList([self._make_layer(block, 384, layers[3], stride=2) for _ in range(num_particle)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.linears = nn.ModuleList([NormedLinear(384*4, num_class) for _ in range(num_particle)])

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride!=1 or self.inplanes!=planes*4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*4),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes*4
        for i in range(1,blocks):
            layers.append(block(self.next_inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        self.logits = []

        for i in range(self.num_particle):
            xi = self.layer4s[i](x)
            xi = self.avgpool(xi)
            xi = xi.view(xi.size(0),-1)
            xi = self.linears[i](xi)
            self.logits.append(xi*30)

        return sum(self.logits)/len(self.logits)
