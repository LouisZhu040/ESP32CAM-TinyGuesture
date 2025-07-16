import torch
from model import MyCNNModel
from data_preprocess import process_data
from test_function import test
from torch.utils.data import TensorDataset, DataLoader
import os
# Confunction matrix for evaluation
from sklearn.metrics import confusion_matrix

# Load the data from data_preprocess section
X_train, X_val, Y_train, Y_val = process_data()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)  
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).squeeze()
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).squeeze()
print('Data processed successfully!')

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  
print('Data loaders created successfully!')

# Initialize the model
model = MyCNNModel()
model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\data_model_train\my_cnn_model.pth"
if os.path.exists(model_path):
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_path))
    print("Model was found successfully!")
else:
    print("No existing model found, starting from scratch...")

print("Model initialized successfully!")


# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Set an early stopping condition
low_loss_counter = 0
loss_threshold = 0.0003
max_low_loss_times = 10
# Set the best validation accuracy
best_val_acc = 0.0 

num_epochs = 50
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0  
    for images, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels) 
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()  
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
    
    # Set the accuracy
    accuracy = test(model, val_loader, criterion)
    model.eval()

    with torch.no_grad():
        val_outputs = model(X_val_tensor)  
        val_preds = val_outputs.argmax(dim=1).cpu().numpy()  
        val_labels = Y_val  
        print("Confusion Matrix：")
        print(confusion_matrix(val_labels, val_preds))

    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with accuracy: {best_val_acc:.2f}%")
    
    # The early stopping condition
    if avg_loss <= loss_threshold:
        low_loss_counter += 1
    else:
        low_loss_counter = 0

    if low_loss_counter > max_low_loss_times:
        print(" Early stopping: loss too low for 5+ epochs.")
        break

    

