import torch
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights


class GCPModel(nn.Module):

    def __init__(self):

        super(GCPModel,self).__init__()

        backbone=resnet18(weights=ResNet18_Weights.DEFAULT)

        self.features=nn.Sequential(*list(backbone.children())[:-1])

        feature_dim=512

        self.coord_head=nn.Sequential(
            nn.Linear(feature_dim,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

        self.shape_head=nn.Sequential(
            nn.Linear(feature_dim,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )


    def forward(self,x):

        features=self.features(x)

        features=features.view(features.size(0),-1)

        coords=self.coord_head(features)

        shape_logits=self.shape_head(features)

        return coords,shape_logits