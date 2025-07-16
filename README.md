# ESP32CAM-TinyGesture
---
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
✅ CNN model defined in PyTorch with detailed comments
✅ Model training and evaluation with accurate metrics
✅ Model save/load capability
✅ Achieved around 90% validation accuracy
```
<img width="442" height="113" alt="12b181a731703fb0b4025e7673bce7e3" src="https://github.com/user-attachments/assets/8ec0553c-52c9-4242-a7b2-d96241ec0146" />

```text
🚧 Converting PyTorch model to TensorFlow Lite (TFLite)
🚧 Flashlight activation logic based on gesture prediction
🚧 Integrate gesture recognizer into ESP32-CAM firmware
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
|   └── train&test.py         # Trains the CNN
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

### 6. Train&Test&Save Model

```bash
python train&test.py
```
Accuracy will be printed.


### 7. Auto Load if Exists

```python
import os
model_path = 'saved_models/my_cnn_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
```

### 8. Convert to TinyML (TODO)

```bash
# Export to ONNX
torch.onnx.export(model, example_input, 'model.onnx')

# Convert to TFLite (via tf-onnx or TFLite converter)
```
---
---

## 🤝 Contributing

```text
Pull requests welcome! Fork, clone, branch, PR.
```
🔖 About the Old_version Branch
- The Old_version branch contains valuable reference material that complements this project. It includes:

- My initial implementations before refactoring

- Detailed comments explaining my understanding of CNN architectures

- Insights on training and testing procedures, including debugging tips and accuracy calculations

- Step-by-step thought processes that led to the current improved codebase

- I believe this branch can be very helpful for visitors who want to learn from the development journey and understand the underlying concepts better.

---
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
