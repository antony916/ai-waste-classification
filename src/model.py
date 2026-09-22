import torch.nn as nn
from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights

def build_model(num_classes):
    m=mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,num_classes)
    return m
