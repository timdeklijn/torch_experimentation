import torch
import torch.nn as nn
from torchsummary import summary

from image import load_and_resize_image


class VggTim(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        # Convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # Convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Pool layers
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Output
        self.output = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, num_classes),
            # Softmax, set dimension along which softmax will be computed
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # go to 1D tensor
        x = self.output(x)
        return x


if __name__ == "__main__":
    # Load the model onto a CPU
    vgg_model = VggTim(100).to(torch.device("cpu"))
    summary(vgg_model, (3, 224, 224))

    # Example of doing inference
    img = torch.tensor([load_and_resize_image("cat.jpg")])
    prediction = vgg_model(img)
    print("Prediction Shape:", prediction.size())
