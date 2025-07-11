# ESP32CAM-TinyGesture 

**ESP32-CAM Image Capture & Gesture Collection Platform**  
Captures images via a browser interface and uploads them to a Flask server—designed for collecting gesture data as a foundation for on-device TinyML model deployment.

---
## Tech Stack:
### ESP32-CAM, Arduino, Python, Flask, TinyML
---

## Why This Project Matters

Nighttime selfies are often underexposed because users forget to turn on the flash. This tool collects gesture-trigger images(✌) to train a TinyML model, with the goal of enabling local gesture-driven flash control—enhancing user experience with low-cost hardware.

Additionally, this project utilizes the low-cost ESP32-CAM hardware to develop an edge AI system for real-time gesture recognition. Using Flask as a lightweight backend, it collects image data efficiently for model training. The ultimate goal is to deploy TinyML models on the device, enabling offline gesture-controlled flashlight functionality.

---

## ✅ Features

**✅ Completed**
- ESP32-CAM capture photos and upload them to a browser control page
- On-click photo capture uploads JPEG via HTTP POST
- Flask server saves files

**🚧 Planned**
- Image dataset collection for a specific gesture
- Train the model to detect “✌” gesture
- Convert to TensorFlow Lite/TinyML
- Integrate the model on ESP32-CAM for offline gesture-controlled flash

---

## 🚀 How to use it

- (1) Open the PlatformIO and prepare to flash the ESP32CAM
- (2) Click the "upload" button from PlatformIO to get started
- (3) Use the following code to monitor the COM port(To be careful, # means the number of the COM port)
~~~bash
pio device monitor --port COM# --baud 115200 --rts 0 --dtr 0
~~~
- (4) Press the reset button on ESP32CAM
- (5) There is an IP for you to check the web that ESP32CAM made
- (6) Run the file named "FlaskSetup.py" to set up your Flask server
- (7) Press "Take Photo" to enable the capture function
- (8) After waiting for a while, the captured photo would be shown on your screen
- (9) The Flask will receive the data of the photo and save it to your local device
- (10) Collecting all the data and looping them in data_preprocesss.py, which is responsible for data processing (normalizing construct array...)
- (11) After gaining the training data and test data, we can get started on training our model.

---
## 💕 Welcome your contribution!!!
**Contributions are welcome! Please open an issue or submit a pull request.**

----
## 😀 License
This project is licensed under the MIT License.

----
## How can you contact me?
E-mail: zhuyiming040@gmail.com



