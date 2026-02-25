import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

if __name__ == "__main__":
    model = get_model()
    print("Model Architecture Summary:")
    print(f"Input channels: {model.conv1.in_channels}")
    print(f"Output classes: {model.fc.out_features}")
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
