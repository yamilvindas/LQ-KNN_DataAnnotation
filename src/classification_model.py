#!/usr/bin/env python3
"""
    This cdode defines the classification model used to evaluate our label
    propagation method through a classification task. It also implements
    Robust Loss functions used to partially compensate the label-noise
    introduce by automatic label propagation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

#==============================================================================#
#=============================Robust Loss Functions=============================#
#==============================================================================#
class SCELoss(torch.nn.Module):
    """
        Symmetric Cross Entropy Loss from https://arxiv.org/abs/1908.06112
    """
    def __init__(self, alpha, beta, A, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0) # modifie les valeurs input = [1e-8, 1.2,0.5] ===> output = [1e-7, 1, 0.5]
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device) # créer un vecteur one hot
        label_one_hot = torch.clamp(label_one_hot, min=np.exp(self.A), max=1.0) # pour éviter d'avoir log(0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class GeneralizedCrossEntropy(torch.nn.Module):
    """
        Generalized Cross Entropy (GCE) https://arxiv.org/abs/1805.07836
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-9
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon

        loss = (1 - p ** self.q) / self.q

        return torch.mean(loss)

#==============================================================================#
#=============================Classification Model=============================#
#==============================================================================#
class MnistClassificationModel(nn.Module):
    def __init__(self):
        super(MnistClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Classification block
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = F.log_softmax(x, dim=1)

        return output


#==============================================================================#
#==============================================================================#
#==============================================================================#
def main():
    # Creating the model
    model = MnistClassificationModel()
    nb_channels, h_in, w_in = 1, 20, 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), (nb_channels, w_in, h_in))

if __name__=="__main__":
    main()
