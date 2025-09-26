import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import EfficientNet_V2_M_Weights, ResNet50_Weights, VGG16_Weights

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        target = F.one_hot(target, num_classes=pred.size(-1))
        target = target.float()
        target = (1 - self.smoothing) * target + self.smoothing / pred.size(-1)
        log_pred = F.log_softmax(pred, dim=self.dim)
        loss = nn.KLDivLoss(reduction='batchmean')(log_pred, target)
        return loss

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # EfficientNet V2 Medium (por defecto)
        self.cnn = torchvision.models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        
        # Opcional: otras arquitecturas
        # self.cnn = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.cnn = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        
        for param in self.cnn.parameters():
            param.requires_grad = True
            
        # Clasificador para EfficientNet
        self.cnn.classifier = nn.Sequential(
            nn.Linear(self.cnn.classifier[1].in_features, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 3),     
        )
        
        # Clasificador para ResNet (descomenta si usas ResNet)
        # self.cnn.fc = nn.Sequential(
        #     nn.Linear(self.cnn.fc.in_features, 512),
        #     nn.Dropout(p=0.2),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 64),
        #     nn.Linear(64, 3)  # Cambiado de 4 a 3 clases
        # )
        
        # Clasificador para VGG (descomenta si usas VGG)
        # self.cnn.classifier = nn.Sequential(
        #     nn.Linear(25088, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(512, 128),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128, 64),
        #     nn.Linear(64, 3)  # Cambiado de 4 a 3 clases
        # )
        
    def forward(self, img):
        output = self.cnn(img)
        return output