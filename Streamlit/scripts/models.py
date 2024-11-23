import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import copy


# model
# baseline
class BaselineEfficientNet(nn.Module):
    def __init__(self, num_classes=6):
        super(BaselineEfficientNet, self).__init__()
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

# class Model(nn.Module):
#     def __init__(self, num_classes=6):
#         super(Model, self).__init__()
#         self.feature_extractor = models.resnet50(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
#         self.classifier = models.efficientnet_b0(pretrained=True)
#         num_features = self.classifier.classifier[1].in_features
#         self.classifier.classifier = nn.Sequential(
#             nn.Dropout(p=0.4),
#             nn.Linear(num_features, num_classes)
#         )

#     def forward(self, x):
#         with torch.no_grad():
#             features = self.feature_extractor(x)
#             features = features.view(features.size(0), -1)
#             outputs = self.classifier(features)
#         return outputs

# for testing purpose we will use the pretrained resnet50 model to extract features and use the features to train the model
class Model(nn.Module):
    def __init__(self, num_classes=6):
        super(Model, self).__init__()
        
        # Use ResNet50 as feature extractor
        self.feature_extractor = models.resnet50(pretrained=True)

        # Remove the last fully connected layer and add a new classifier layer
        num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # Forward pass through ResNet
        return self.feature_extractor(x)
