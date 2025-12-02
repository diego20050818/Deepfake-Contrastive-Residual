import torch
import torch.nn as nn
import clip


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Model(nn.Module):
    def __init__(self, 
                 backbone=("ViT-B/32", 768),
                 nproj=2,
                 proj_dim=512,
                 device='cuda'):
        super().__init__()
        self.device = device
        # Load and freeze CLIP
    
        self.clip, self.preprocess = clip.load(backbone[0], device=device)
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ] # type: ignore
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ] # type: ignore
            )
        self.proj2 = nn.Sequential(*proj2_layers)

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]
        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)
        
        return z  # 只返回特征，不返回预测值

# 然后修改 FeatureExtractor.py 的测试部分
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())

    
    # 使用 RINE 模型作为特征提取器
    backbone = ("ViT-B/32", 768)
    # model = RINEFeatureExtractor(backbone=backbone, device='cpu')  # 使用CPU避免CUDA问题
    
    from utils.transforms import transforms_train
    img = 'dataset/own/01_fake_PS.jpg'
    from torchvision import transforms
    from PIL import Image
    img = Image.open(img).convert('RGB')
    img = transforms_train(img).unsqueeze(0)
    
    feature = model(img)
    print(f"Extracted features shape: {feature.shape}")
    print(f"Features sample values: {feature[0, :10]}")  # 显示前10个特征值
