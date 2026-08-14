import torch
import torch.nn as nn
from config import ModelConfig

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample = None, middel_conve_stride = 1, residual = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = planes, kernel_size = 1, stride = middel_conve_stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels = planes, out_channels = planes, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels = planes, out_channels = planes * 4, kernel_size = 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.residual = residual

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        

        if self.residual:
            if self.downsample is not None:
                x = self.downsample(x)
            out = out + x
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.model == "resnet50":
            self.layers_count = [3, 4, 6, 3]
        elif config.model == "resnet101":
            self.layers_count = [3, 4, 23, 3]
        else:
            self.layers_count = [3, 8, 36, 3]
        
        self.in_planes = 64
        self.model = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = self.in_planes, kernel_size = 7, stride = 2, padding = 3),   
        nn.BatchNorm2d(self.in_planes),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1),

        self._make_layers(self.layers_count[0], planes = 64, stride = 1),
        self._make_layers(self.layers_count[1], planes = 128, stride = 2),
        self._make_layers(self.layers_count[2], planes = 256, stride = 2),
        self._make_layers(self.layers_count[3], planes = 512, stride = 2),
        nn.AdaptiveAvgPool2d((1, 1)))
        self.predictor = nn.Linear(2048, self.config.num_classes)


    def forward(self, x):
        x = self.model(x).flatten(1)
        return self.predictor(x)

    def _make_layers(self, num_residual_block, planes, stride):
        downsample = None
        layers = nn.ModuleList()

        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes * 4, 1, stride = stride),
                            nn.BatchNorm2d(planes*4))
        layers.append(ResidualBlock(in_planes = self.in_planes,
                        planes = planes, 
                        downsample = downsample,
                        middel_conve_stride = stride,
                        residual = self.config.residual))
        self.in_planes = planes * 4
        for _ in range(num_residual_block -1):
            layers.append(ResidualBlock(in_planes = self.in_planes, planes = planes, residual = self.config.residual))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    config = ModelConfig()
    rand = torch.rand(1, 3, 224, 224)
    model = ResNet(config)
    print(model(rand).shape)
    print(sum([p.numel() for p in model.parameters()]))

