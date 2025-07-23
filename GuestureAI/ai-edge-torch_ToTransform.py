# Install dependencies
# pip install -q ai-edge-torch torch

# Upload the .pth model file
from google.colab import files
uploaded = files.upload()  # Please upload my_cnn_model.pth

# Define the model (must match the training structure exactly)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 7 * 10, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 7 * 10)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load weights
model = MyCNNModel()
model.load_state_dict(torch.load("my_cnn_model.pth", map_location="cpu"))
model.eval()

# Dummy input
dummy = torch.randn(1, 1, 120, 160)

# Convert with ai-edge-torch
import ai_edge_torch
edge_model = ai_edge_torch.convert(model, (dummy,))

# Export as TFLite model
edge_model.export("/content/model.tflite")

# Download the model
files.download("model.tflite")
print("🎉 Export and download completed!")
