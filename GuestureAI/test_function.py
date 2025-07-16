import torch
from model import MyCNNModel
from data_preprocess import process_data
from torch.utils.data import TensorDataset, DataLoader

def test(model, val_loader, criterio):
    # Start testing the model
    model.eval()  # set the model to evaluation mode
    correct = 0     # Initialize the number of correct predictions
    total = 0       # Initialize the total number of samples

    # Disable gradient calculation for validation to save memory and computation
    # This is useful because we don't need to compute gradients during validation
    with torch.no_grad():  
        # Loop through each batch of validation data
        for images, labels in val_loader:
            outputs = model(images)             # Forward pass: compute the model's predictions
            # outputs.data contains the raw output from the model
            # torch.max() returns the maximum value and its index along the specified dimension
            # Because the output is a 2D tensor, we use dim=1 to find the maximum value along the class dimension
            # we don't need the maximum value as we only want to know which class the model chooses, so we use _ to ignore it
            _, predicted = torch.max(outputs.data, 1)  # _, predicted is named "unpacking", which means we only need the second value (the predicted class)

            total += labels.size(0)                          # The total number of samples
            correct += (predicted == labels).sum().item()    # Count the number of correct predictions

    # Print the accuracy of the model on the validation set
    accuracy = 100 * correct / total
    print(f'Accuracy on validation set: {accuracy:.2f}%')
    return accuracy

