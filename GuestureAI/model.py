import torch.nn as nn
import torch.nn.functional as F

class MyCNNModel(nn.Module):
    def __init__(self):
        super(MyCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_size = 64 * 15 * 20
        
        self.fc1 = nn.Linear(self.flatten_size, 64)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        
        x = F.relu(x)
        
        x = self.pool1(x)
        
        x = self.conv2(x)
        
        x = F.relu(x)
        
        x = self.pool2(x)
        
        x = self.conv3(x)
        
        x = F.relu(x)
        
        x = self.pool3(x)
        
        x = x.reshape(-1, self.flatten_size)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        return x
