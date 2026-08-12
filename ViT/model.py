import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            )
        self.linear = nn.Sequential(
            nn.Linear(in_features = 6400, out_features = 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
                        
    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.linear(x)
        
    def num_parameters(self):
        count = 0
        for layer in self.parameters():
            count += layer.numel()
        return count

if __name__ == "__main__":
    data = torch.rand((10, 3, 224, 224))
    model = AlexNet()
    print(model(data).shape)
    print(model.num_parameters())

