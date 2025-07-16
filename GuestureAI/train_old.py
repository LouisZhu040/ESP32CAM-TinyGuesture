# import our model and our data-processor
import torch
from model import MyCNNModel
from data_preprocess import process_data
# TensorDataset can match the data and labels together
# Dataloader can load the data in batches
from torch.utils.data import TensorDataset, DataLoader
import os

# Load the data from data_preprocess section
X_train, _ , Y_train, _ = process_data()

# Because the Numpy cannot be directly used by CNN, transformation into tensor is required
# Since the images are normalized to [0, 1], we use float32 to represent the pixel values
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)     
# Convert the labels to 1D tensor by using squeeze() to remove the extra dimension
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).squeeze()  # troch.long is used for integer labels
print('data processed successfully!')

# Create TensorDataset to combine the training data and labels
# This will allow us to easily access the data and labels together during training
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# Create DataLoader to load the training data in batches 
# batch_size=32 means we will use 32 images in each batch, shuffle=True means the data will be randomly shuffled before each epoch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model
model = MyCNNModel()

# Save the model to a file if it does not exist
model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\data_model_train\my_cnn_model.pth"

# Load the previous model if it exists
if os.path.exists(model_path):
    print("Loading existing model...")
    # Load the model's state_dict from the file
    model.load_state_dict(torch.load(model_path))
    print("Model was found successfully!")
else:
    print("No existing model found, starting from scratch...")

print("Model initialized successfully!")

# Define the loss function and optimizer
# CrossEntropyLoss's funnction is to distinguish the class and cal the accuracy
criterion = torch.nn.CrossEntropyLoss()
# Adam optimizer is used to update the model's parameters
# lr=0.001 is the learning rate, which controls how much to change the model's parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set an early stopping condition
low_loss_counter = 0
loss_threshold = 0.0003
max_low_loss_times = 5 

# Training the model
num_epochs = 10  # Number of epochs to train the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize the running loss for this epoch

    # Loop through each batch of data
    for images, labels in train_loader:
        # Change the shape again from (batch_size, height, width, channels) to (batch_size, channels, height, width)
        # We only need to change the shape of the images, not the accuracy labels
        images = images.permute(0, 3, 1, 2) 
        optimizer.zero_grad()  # Zero the gradients before the backward pass
        outputs = model(images)  # Forward pass: compute the model's predictions
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update the model's parameters
        running_loss += loss.item()  # Accumulate the loss
    avg_loss = running_loss / len(train_loader) # Accumulate the loss for this batch
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
    # The early stopping condition
    if avg_loss <= loss_threshold:
        low_loss_counter += 1
    else:
        low_loss_counter = 0 

    if low_loss_counter >= max_low_loss_times:
        print(f" Loss continutely loss {max_low_loss_times} times ≤ {loss_threshold}，stop training.")
        break
    

# Save the trained model
torch.save(model.state_dict(), model_path)