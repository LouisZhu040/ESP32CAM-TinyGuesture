import onnx
from onnx_tf.backend import prepare

onnx_model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\data_model_train\gesture_model.onnx"
# Load the ONNX model
onnx_model = onnx.load(onnx_model_path) 
tf_rep = prepare(onnx_model)

guess_model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\data_model_train\gesture_tf_model"
# Export to TensorFlow SavedModel format
tf_rep.export_graph(guess_model_path)  
print("Successfully converted to TensorFlow SavedModel format!")
