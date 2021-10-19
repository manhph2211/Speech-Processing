from collections import OrderedDict
import torch
import torch.nn as nn 


class VGGVox(nn.Module):
    def __init__(self, n_classes=512):
        super(VGGVox, self).__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))),
            ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))),
            ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu5', nn.ReLU()),
            ('pool5', nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))),
            ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(1, 2), stride=(1, 1))),
            ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
            ('relu6', nn.ReLU()),
            ('pool6', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten())]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(1024, n_classes))]))

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_tensor):
        output_tensor = self.features(input_tensor)
        output_tensor = self.classifier(output_tensor)
        return output_tensor