import torch
from torch import nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) # 这里没有传 batch size
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 10, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2240, 512) # 80 x 240 x 10 = 2240
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 238)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.avg_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 2240) # 80 x 240 x 10 = 2240
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 7, 34)
        x = F.softmax(x, dim=2)
        x = x.view(-1, 238)
        return x

# model = Net()
# out = model(torch.rand([1,3,80,240]))
# print(out.shape)
