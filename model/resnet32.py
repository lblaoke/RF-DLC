import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1,1).renorm_(2,1,1e-5).mul_(1e5)
    def forward(self,x):
        return F.normalize(x,dim=1).mm(F.normalize(self.weight,dim=0))

class BasicBlock(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = lambda x: x
        if stride!=1 or in_planes!=planes:
            self.planes = planes
            self.in_planes = in_planes
            self.shortcut = lambda x: F.pad(x[:,:,::2,::2],(0,0,0,0,(planes-in_planes)//2,(planes-in_planes)//2),"constant",0)
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))+self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet32(nn.Module):
    def __init__(self,block,num_blocks,num_particle,num_class):
        super(ResNet32,self).__init__()

        self.in_planes = 16
        self.num_class = num_class
        self.num_particle = num_particle

        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1s = nn.ModuleList([self._make_layer(block,16,num_blocks[0],stride=1) for _ in range(num_particle)])
        self.in_planes = self.next_in_planes

        self.layer2s = nn.ModuleList([self._make_layer(block,32,num_blocks[1],stride=2) for _ in range(num_particle)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block,64,num_blocks[2],stride=2) for _ in range(num_particle)])
        self.in_planes = self.next_in_planes

        self.linears = nn.ModuleList([NormedLinear(64,num_class) for _ in range(num_particle)])

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes
        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not module.weight.requires_grad:
                    module.eval()

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        self.logits = []

        for i in range(self.num_particle):
            xi = self.layer1s[i](x)
            xi = self.layer2s[i](xi)
            xi = self.layer3s[i](xi)
            xi = F.avg_pool2d(xi,xi.shape[3])
            xi = xi.flatten(1)
            xi = self.linears[i](xi)
            self.logits.append(xi*30)

        return sum(self.logits)/len(self.logits)
