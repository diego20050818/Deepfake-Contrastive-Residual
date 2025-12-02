"""
特征提取模块
在这里会定义相关的骨干网络
"""
import sys
import os
sys.path.append(os.getcwd()) 
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
from loguru import logger
import module

def get_backbone(model_name:str='rine',load_from:Literal['remote','local']='remote',**config:Optional[dict]):
    base_model = None
    try:
        if load_from == 'remote':
            base_model = getattr(models, model_name)(pretrained=True,**config )
            # ResNet系列
            if 'resnet' in model_name:
                base_model = nn.Sequential(*list(base_model.children())[:-1])
            # DenseNet系列
            elif 'densenet' in model_name:
                base_model.classifier = nn.Identity()
            # VGG/AlexNet系列
            elif 'vgg' in model_name or 'alexnet' in model_name:
                base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
            # 默认处理
            else:
                base_model = nn.Sequential(*list(base_model.children())[:-1])
                
        if load_from == 'local':
            base_model = getattr(module,model_name)(**config) # TODO 添加传参
    except Exception as e:
        logger.error(f'fail to load base model:{e}')
    return base_model

class FeatureExtractor(nn.Module):
    def __init__(self,base_model:str,
                 load_from:Literal['remote','local']='remote',
                 output_dim=2048,
                 **config) -> None:
        super().__init__()
        self.backbone = get_backbone(model_name=base_model,load_from=load_from,**config)
        if self.backbone == None:
            raise KeyError("backbone == none,please check")
        
        self.output_dim = output_dim

    def forward(self,x):
        # [b,3,224,224] -> [b,2048,1,1]
        features = self.backbone(x) # type:ignore
        # [b,2048,1,1] -> [b,2048]
        return torch.flatten(features, 1)
        

if __name__ == '__main__':
    # basemodel = get_backbone()
    # device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = FeatureExtractor(base_model='rine',load_from='local',
                             backbone=("ViT-B/32", 768),
                             nproj=2,
                             proj_dim=512,
                             device='cpu')
    
    from utils.transforms import transforms_train
    img = 'dataset/own/01_fake_PS.jpg'

    from PIL import Image
    img = Image.open(img).convert('RGB')
    img = transforms_train(img).unsqueeze(0) # type: ignore
    # img.to(device)
    feature = model(img)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # print(feature.shape)
    feature = feature.squeeze(0)
    heatmap = sns.violinplot(feature.detach().numpy() )
    plt.savefig('box.png') 
    print(feature)
