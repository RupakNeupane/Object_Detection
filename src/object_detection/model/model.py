import torch.nn as nn
from torchvision import models
import torch

from src.object_detection.model.resnet50 import get_backbone


def get_model(backbone,n_classes,label2target):
    background_class = label2target['background']

    class RCNN(nn.Module):
        def __init__(self, backbone, n_classes):
            super().__init__()
            self.backbone = backbone
            self.n_classes = n_classes
            
            self.classification_head = nn.Linear(2048,n_classes)

            self.bbox_localization_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 4),
                nn.Tanh()
            )
            
            
            
            self.classification_loss = nn.CrossEntropyLoss()
            self.localization_loss = nn.L1Loss()
            self.lmbda = 10.0 #priotizes localization loss over classification loss
            
            
            
        def forward(self, inputs):
            feat = self.backbone(inputs)
            
            cls_score = self.classification_head(feat)
            
            deltas = self.bbox_localization_head(feat)
            
            return cls_score, deltas
        
        
        def calculate_loss(self, _labels, _deltas, actual_labels, actual_deltas):
            
            #classification loss   
            classification_loss = self.classification_loss(_labels, actual_labels)
            
            #localization loss
            ix = torch.where(actual_labels != background_class)[0]
            _deltas = _deltas[ix]
            actual_deltas = actual_deltas[ix]
            
            if (len(ix)>0):
                localization_loss = self.localization_loss(_deltas, actual_deltas)
            else:
                localization_loss = torch.tensor(0)
                
            total_loss = classification_loss + self.lmbda * localization_loss
            
            return total_loss, classification_loss, localization_loss
        
    return RCNN(backbone,n_classes)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rcnn = get_model(get_backbone(device), 3).to(device=device)

    print(rcnn)