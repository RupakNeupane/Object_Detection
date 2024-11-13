import torch.nn as nn
from torchvision import models
import torch

def get_backbone(device):
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Sequential()

    for param in backbone.parameters():
        param.requires_grad = False
    
    return backbone.to(device=device) 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_backbone(device=device)
    print(model)