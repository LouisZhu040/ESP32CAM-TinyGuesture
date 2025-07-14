# ESP32CAM-TinyGesture

A lightweight **gesture data collection and recognition platform** built with **ESP32-CAM** and **Flask**, designed to capture images through a browser interface, upload them for training, and eventually run **real-time, offline gesture recognition** with **TinyML**. For example: ✌️ gesture to trigger the flash in dark environments.

---

## 🚀 Tech Stack

```text
- Hardware: ESP32-CAM
- Programming: Arduino C++, Python
- Backend: Flask Server
- AI/ML: PyTorch for training → TensorFlow Lite / TinyML for deployment
```

---

## 🎯 Project Goals

Most users forget to turn on flash when taking selfies in the dark. This project solves that by training a model to recognize the ✌️ gesture, which automatically activates the flash.

```text
✅ Capture and label gesture data using ESP32-CAM + browser
✅ Preprocess and normalize data for training
✅ Train a CNN (Convolutional Neural Network) model in PyTorch
✅ Evaluate model performance
✅ Save and load model
🚧 Convert the trained model to TensorFlow Lite
🚧 Deploy the model on ESP32-CAM for offline, real-time gesture recognition
```

---

## ✅ Completed Features

```text
✅ ESP32-CAM live photo capture via web interface
✅ HTTP POST to Flask backend
✅ Flask server receives and saves image data
✅ Data preprocessing script (grayscale, normalize, reshape)
✅ CNN model defined in PyTorch
✅ Model training and evaluation
✅ Model save/load capability
```

## 🚧 Coming Soon

```text
🚧 ✌ Gesture dataset expansion
🚧 Convert PyTorch model to TensorFlow Lite (TFLite)
🚧 Flashlight activation logic based on gesture prediction
🚧 Integrate gesture recognizer into ESP32-CAM firmware
🚧 Improve FPS and responsiveness on edge
```

---

## 📁 Project Structure

```text
ESP32CAM-TinyGesture/
├── ESP32CAM/              # Arduino sketch for camera control + web UI
├── GuestureAI/
│   |── FlaskSetup.py      # Flask backend to receive and store uploaded images
|   └── data_preprocess.py     # Data normalization, grayscale conversion, reshape
|   └── model.py               # CNN architecture in PyTorch
|   └── train&model.py         # Trains the CNN
├── Photos/
│   ├── ye/                # ✌ gesture samples (label: 1)
│   └── noye/              # Neutral/other samples (label: 0)
└── README.md              # You are here :)
```

---

## 🧪 How to Use It

### 1. Setup ESP32-CAM

```bash
pio run -t upload
pio device monitor --port COM# --baud 115200 --rts 0 --dtr 0
```

Get IP address from Serial Monitor.

### 2. Run Flask Server

```bash
python FlaskServer/FlaskSetup.py
```

### 3. Capture Image from Web

Open browser → enter ESP32 IP → Click `Take Photo`

### 4. Save Images in Flask Server

Images saved in:

```text
Photos/ye/
Photos/noye/
```

### 5. Preprocess Images

```bash
python data_preprocess.py
```

Processes images: grayscale → normalize → reshape → split

### 6. Train Model

```bash
python train_model.py
```

Model:

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 7. Test Model

```bash
python test_model.py
```

Accuracy will be printed.

### 8. Save/Load Model

```python
# Save model
torch.save(model.state_dict(), 'saved_models/my_cnn_model.pth')

# Load model
model.load_state_dict(torch.load('saved_models/my_cnn_model.pth'))
```

### 9. Auto Load if Exists

```python
import os
model_path = 'saved_models/my_cnn_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
```

### 10. Convert to TinyML (TODO)

```bash
# Export to ONNX
torch.onnx.export(model, example_input, 'model.onnx')

# Convert to TFLite (via tf-onnx or TFLite converter)
```

---

## 🤝 Contributing

```text
Pull requests welcome! Fork, clone, branch, PR.
```

---

## 📫 Contact

```text
Email: zhuyiming040@gmail.com
```

---

## 📄 License

```text
MIT License. Free to use and modify.
```
