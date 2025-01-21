from torch import nn
from . import resnet32
from . import resnet50

class BaseModel(nn.Module):
    requires_target = False

    def __init__(self,num_class,backbone_class=None):
        super().__init__()
        if backbone_class is not None:
            self.backbone = backbone_class(num_class)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self,x):
        return self.backbone(x)

class ResNet32Cifar(BaseModel):
    def __init__(self,num_class,num_particle=1,**kwargs):
        super().__init__(num_class,None)
        self.backbone = resnet32.ResNet32(
            resnet32.BasicBlock         ,
            [5,5,5]                     ,
            num_class    = num_class    ,
            num_particle = num_particle ,
            **kwargs
        )

class ResNet50Imagenet(BaseModel):
    def __init__(self,num_class,layer3_output_dim=None,layer4_output_dim=None,num_particle=1,**kwargs):
        super().__init__(num_class,None)
        self.backbone = resnet50.ResNet50(
            resnet50.Bottleneck                 ,
            [3,4,6,3]                           ,
            num_class         = num_class       ,
            num_particle      = num_particle    ,
            **kwargs
        )

class ResNet50iNaturalist(BaseModel):
    def __init__(self,num_class,layer3_output_dim=None,layer4_output_dim=None,num_particle=1,**kwargs):
        super().__init__(num_class,None)
        self.backbone = resnet50.ResNet50slim(
            resnet50.Bottleneck                 ,
            [3,4,6,3]                           ,
            num_class         = num_class       ,
            num_particle      = num_particle    ,
            **kwargs
        )

class ResNet32DermaMNIST(BaseModel):
    def __init__(self,num_class,num_particle=1,**kwargs):
        super().__init__(num_class,None)
        self.backbone = resnet32.ResNet32(
            resnet32.BasicBlock         ,
            [5,5,5]                     ,
            num_class    = num_class    ,
            num_particle = num_particle ,
            **kwargs
        )

if __name__ == "__main__":
    from torchinfo import summary
    model = ResNet32Cifar(100, num_particle=3)
    summary(model, input_size=(128, 3, 32, 32))
