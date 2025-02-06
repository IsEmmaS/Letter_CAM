import torch
import torch.nn as nn


class letterClassifier(nn.Module):

    def __init__(self) -> None:
        super(letterClassifier, self).__init__()

        self.convBlock = lambda in_planes, out_planes: nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_planes),
            nn.MaxPool2d(2),
        )


        self.model = nn.Sequential(
            self.convBlock(3, 64),  # 输出: [64, 32, 32]
            self.convBlock(64, 128),  # 输出: [128, 16, 16]
            self.convBlock(128, 256),  # 输出: [256, 8, 8]
            self.convBlock(256, 512),  # 输出: [512, 4, 4]
            self.convBlock(512, 64),  # 输出: [64, 2, 2]
            nn.Flatten(),  # 展平: [64, 2, 2] → [256]
            nn.Linear(256, 256),  # 修改输入维度为 256
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, 26),
        )
        self.loss_fn = nn.CrossEntropyLoss()

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


if __name__ == "__main__":
    model = letterClassifier()
    x = torch.randn(2, 3, 64, 64)
    output = model(x)
    print(output.shape)
