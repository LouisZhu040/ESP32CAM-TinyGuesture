// include necessary libraries
#include <Arduino.h>
#include <camera_pins.h>
#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
#include <HTTPClient.h>
// Import the model and TensorFlow Lite libraries
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

// Define camera pins for AI Thinker model
#define CAMERA_MODEL_AI_THINKER

// Define the GPIO pins for the LED
const int LED_PIN = 4;
const char* ssid = "iPhone 16 Pro Max";   // Please replace with your WiFi SSID if you want to use WiFi
const char* password = "20070121Louis";   // Please replace with your WiFi password if you want to use WiFi

// Set fb to null pointer(Global variable)
camera_fb_t *fb = NULL;

WebServer server(80);
// Function to construct the HTTP response for the photo stream
void constructHtml() {
  String html = R"rawliteral(
    <!DOCTYPE html>
<html>
  <head>
    <title>ESP32-CAM Photo</title>
    <style>
      body {
        margin: 0;
        padding: 0;
        font-family: "Segoe UI", sans-serif;
        background: #f5f7fa;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        min-height: 100vh;
      }

      h2 {
        margin-top: 30px;
        color: #333;
      }

      .camera-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
        padding: 20px;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      }

      img {
        width: 100%;
        max-width: 480px;
        border-radius: 10px;
        margin-bottom: 20px;
      }

      .button {
        padding: 14px 28px;
        font-size: 16px;
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
      }

      .button:hover {
        background-color: #1976D2;
      }
    </style>
  </head>
  <body>
    <h2>ESP32-CAM Web</h2>
    <div class="camera-container">
      <img id="photo" src="/photo" alt="Live Snapshot">
      <button class="button" onclick="takePhoto()">Take Photo</button>
    </div>

    <script>
      function takePhoto() {
        const img = document.getElementById('photo');
        img.src = '/photo?t=' + new Date().getTime(); // 强制刷新
      }
    </script>
  </body>
</html>
)rawliteral";
  server.send(200, "text/html", html);
}

// Function to capture a photo
void capturePhoto() {
  // Start to catch the photo
  Serial.println("Capturing photo...");
  fb = esp_camera_fb_get();
  Serial.println("Photo captured");
  // Check if the frame buffer was acquired successfully
  if (!fb) {
    Serial.println("Failed to capture image");
    return;
  }
}

// Function to submit the captured photo to the connected HTTP client (e.g., browser)
// This function sends the captured photo as a JPEG image over HTTP to the client
void submitPhoto() {
  // Set the HTTP response header to indicate the content type is a JPEG image
  server.sendHeader("Content-Type", "image/jpeg");
  // Set the HTTP response header to specify the length of the content (image size in bytes)
  server.sendHeader("Content-Length", String(fb->len));

  // Get the currently connected client (e.g., browser)
  WiFiClient client = server.client();

  // Check if the client is still connected
  if (client.connected()) {
    // Write the image data stored in the frame buffer to the client
    client.write(fb->buf, fb->len);

    // Check again if the client is still connected after sending data
    // Because sending large data may cause disconnection
    if (!client.connected()) {
      // If the client disconnected during data transfer, log a warning
      Serial.println("Client disconnected before sending photo");
    }

    // Send HTTP status code 200 to indicate the request was successful
    // This must be sent after the data, otherwise some clients may report errors
    server.send(200);

    // Log success message after sending the photo
    Serial.println("Photo sent successfully");
  } else {
    Serial.println("Failed to send photo, client not connected");
  }
}

// As we need to training the ESP32-CAM, we need to save the photo to the server
void savePhoto() {
  // Set a variable to calculate the times of re-uploading the photo
  int saveAttempts = 0; 
  // Set a variable to check if the photo is saved successfully
  bool saveSuccess = false;

  // Set a loop to retry saving the photo up to 3 times
  while(saveAttempts <= 3 && saveSuccess == false) {

    // Create an HTTP client to post the photo to a server
    HTTPClient http;
    Serial.println("Uploading photo to server...");

    // Specify the URL of the server where the photo will be uploaded
    http.begin("http://172.20.10.2:5000/upload"); // Replace with your server URL

    // Based on my testing, the ESP32-CAM's timeout is very short
    // So I enlongate the timeout to 15 seconds
    http.setTimeout(15000); // Set timeout to 15 seconds
    
    // Set the content type to JPEG
    http.addHeader("Content-Type", "image/jpeg");

    // Send the photo data as a POST request
    // The varible httpResponseCode will store the response code from the server. "200" means success
    int httpResponseCode = http.POST(fb->buf, fb->len);

    // Check if the HTTP request was successful
    if (httpResponseCode > 0) {
      // Log the response code from the server
      Serial.printf("Photo uploaded successfully, response code: %d\n", httpResponseCode);
      saveSuccess = true; 
      // End the HTTP connection
      http.end();
      // Turn on the LED to indicate the photo is saved successfully
      digitalWrite(LED_PIN, HIGH);
      delay(1000); // Keep the LED on for 1 second
      digitalWrite(LED_PIN, LOW); // Turn off the LED
      return;
    }else {
      // Log an error message if the upload failed
      Serial.printf("Failed to upload photo, error: %s\n", http.errorToString(httpResponseCode).c_str());
      // Increment the saveAttempts counter
      saveAttempts++;
      Serial.printf("Retrying to upload photo, attempt %d\n", saveAttempts);
      // End the HTTP connection
      http.end();
    }
  }
}

// Function to release the photo frame buffer, which frees up the memory used by the captured photo
void releasePhoto() {
  // Release the frame buffer to free up memory
  if (fb) {
    esp_camera_fb_return(fb);
    fb = NULL; // Set fb to NULL after releasing it
    Serial.println("Photo frame buffer released");
  } else {
    Serial.println("No photo frame buffer to release");
  }
}

// An Combined function to handle the photo capture and submission
void handlePhoto() {
  capturePhoto();
  submitPhoto();
  savePhoto();
  releasePhoto();
}

// Setup function to initialize the camera and start the web server
// This function is called once when the ESP32 starts
void setup() {
  Serial.begin(115200);
  //init camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  // set the photo format
  config.pixel_format = PIXFORMAT_JPEG;
  
  // set the frame size and quality
  if(psramFound()){
    Serial.println("PSRAM found - using high resolution");
    config.frame_size = FRAMESIZE_UXGA;    
    config.jpeg_quality = 20;              
    config.fb_count = 1;                   
  } else {
    Serial.println("PSRAM not found - using low resolution");
    config.frame_size = FRAMESIZE_SVGA;    
    config.jpeg_quality = 60;
    config.fb_count = 1;
  }

  //setup the camera
  esp_err_t err = esp_camera_init(&config);
  // Check if the camera was initialized successfully
  if (err == ESP_OK) {
    Serial.println("Camera initialized successfully");
  } else {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.println(WiFi.localIP());

  // Start the web server
  server.begin();
  server.on("/", constructHtml);

  // Handle the photo request
  server.on("/photo", handlePhoto);

  // Set the mode of the LED pin to OUTPUT
  pinMode(LED_PIN, OUTPUT);
  // Turn off the LED initially
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  // Handle client requests
  server.handleClient();
  // Check if the WiFi is still connected
  if (WiFi.status() != WL_CONNECTED) {
    // If WiFi is disconnected, attempt to reconnect
    Serial.println("WiFi disconnected, reconnecting...");
    // Attempt to reconnect to WiFi
    WiFi.reconnect();
    // Wait until the connection is re-established
    Serial.print("Reconnecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    // Report successful reconnection
    Serial.println("\nReconnected to WiFi!");
  }
}