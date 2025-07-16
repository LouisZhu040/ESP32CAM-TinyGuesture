import torch
from model import MyCNNModel


model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\data_model_train\my_cnn_model.pth"
# load the existing model
model = MyCNNModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Create a dummy input for ONNX export
dummy_input = torch.randn(1, 1, 120, 160)

onnx_model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\data_model_train\gesture_model.onnx"
# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print("Exported model to ONNX format successfully at:", onnx_model_path)


