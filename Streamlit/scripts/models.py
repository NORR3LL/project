import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import copy


# model
# baseline
class Baseline(nn.Module):
    def __init__(self, num_classes=5):
        super(Baseline, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False  # freeze
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# for testing purpose we will use the pretrained resnet50 model to extract features and use the features to train the model
class Model(nn.Module):
    def __init__(self, num_classes=5):
        super(Model, self).__init__()
        
        # Use ResNet50 
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
