import torch
import torch.nn as nn


class letterClassifier(nn.Module):

    def __init__(self) -> None:
        super(letterClassifier, self).__init__()

        self.model = nn.Sequential(
            self.convBlock(3, 64),
            self.convBlock(64, 64),
            self.convBlock(64, 128),
            self.convBlock(128, 256),
            self.convBlock(256, 512),
            self.convBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, 52),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def convBlock(in_planes: int, out_planes: int) -> nn.Sequential:
        """
        Convolutional Block
        :param in_planes: int number of input channels
        :param out_planes: int number of output channels
        :return: block with conv layers
        """
        return nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_planes),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        :param x: input tensor
        :return: output tensor
        """
        return self.model(x)

    def compute_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate metrics
        :param preds: tensor output from model
        :param targets: targets from datasets
        :return: A tuple (loss, acc)
        """
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1] == targets).float().mean()

        return loss, acc
