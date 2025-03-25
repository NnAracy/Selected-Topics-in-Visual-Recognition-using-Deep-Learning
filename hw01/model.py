import torch
import torchvision.models as models
import torch.nn as nn

class CustomClassifier(nn.Module):
    def __init__(self, num_ftrs, num_classes, dropout):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def get_model(pretrained_path=None, model_name="resnet101",
              num_classes=100, device="cuda", dropout=None):
    if model_name == "resnet101":
        model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
        )
    elif model_name == "resnext50":
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
    elif model_name == "resnext101":
        model = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
        )

    print(f"running model {model_name}")

    num_ftrs = model.fc.in_features
    if model_name == "resnet101":
        model.fc = CustomClassifier(num_ftrs, 100, dropout)
        print("using custom fc layer")
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)

    model.to(device)
    return model