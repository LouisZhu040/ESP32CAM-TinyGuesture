import torch.nn as nn
import torch.nn.functional as F

# Start to create CNN model
class  MyCNNModel(nn.Module):
    # Define the architecture of the CNN model
    def __init__(self):
        # Initialize the parent class
        super(MyCNNModel, self).__init__()

        # in_channels=1 means the input is a grayscale image
        # out_channels=8 means the first convolutional layer will output 8 feature maps
        # kernel_size=3 means the convolutional filter is 3x3
        # stride=1 means the filter moves one pixel at a time 
        # padding=1 means we add a border of 1 pixel around the input image to keep the output size the same as the input size.
        # This is the first convolutional layer, which will learn 8 different filters to extract
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1) 

        # The first pooling layer: Max pooling with a 2x2 window and a stride of 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # The second convolutional layer, which haves 16 output channels
        # This layer will learn 16 different filters to extract more complex features from the output of the first pooling layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        # The second pooling layer: Max pooling with a 2x2 window and a stride of 2
        # This layer will further reduce the size of the feature maps
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # kernel_size=2 means the pooling window is 2x2 and stride=2 means it moves 2 pixels at a time

        # Because the input image is 120x160, after two convolutional layers and two pooling layers
        # The output size will be reduced to 16x30x40
        # We have 16 feature images after the second convolutional layer
        self.flatten_size = 16 * 30 * 40

        # First fully connected layer: takes the flattened feature vector from the flatten layer 
        # and outputs 64 neurons. This helps to reduce the size and extract key features.
        self.fc1 = nn.Linear(self.flatten_size, 64)  
        # The second fully connected layer: takes the output from the first fully connected layer
        # and outputs 2 neurons, which corresponds to the two classes (Ye and Noye).
        self.fc2 = nn.Linear(64, 2)                    # 再输出2类（可以改为你需要的类别数）

    # Define the forward pass of the model
    # This method defines how the input data flows through the model
    def forward(self, x):
        x = self.conv1(x)           # The first convolutional layer: extracts features from the input image
        # Activation function: ReLU (Rectified Linear Unit) is applied to introduce non-linearity and contain the feature
        x = F.relu(x)               
        x = self.pool1(x)           # Pooling layer 1: reduces the size of the feature maps

        x = self.conv2(x)           # The second convolutional layer: extracts more complex features
        x = F.relu(x)               # Activation function: ReLU is applied again
        x = self.pool2(x)           # Pooling layer 2: further reduces the size of the feature maps

        # .view() is used to reshape the tensor
        # -1 means the size of this dimension will be inferred from the other dimensions
        x = x.view(-1, self.flatten_size)  # Flatten the output into a 1D vector
        x = F.relu(self.fc1(x))            # Fully connected and activation
        x = self.fc2(x)                    # Final output layer: produces the final predictions

        return x


