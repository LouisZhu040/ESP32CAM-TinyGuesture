// include necessary libraries
#include <Arduino.h>
#include <camera_pins.h>
#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
// Define camera pins for AI Thinker model
#define CAMERA_MODEL_AI_THINKER

const char* ssid = "iPhone 16 Pro Max";
const char* password = "20070121Louis";

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

void submitPhoto() {
  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Length", String(fb->len));
  WiFiClient client = server.client();
  if (client.connected()) {
    client.write(fb->buf, fb->len);
      if(client.connected()) {
        // Return the frame buffer to be reused
      esp_camera_fb_return(fb);
      }
      else {
        Serial.println("Client disconnected before sending photo");
      }
    fb = NULL; // Set fb to null pointer after sending
    server.send(200);
    Serial.println("Photo sent successfully");
  }
  else {
    Serial.println("Failed to send photo, client not connected");
  }
}

void handlePhoto() {
  capturePhoto();
  submitPhoto();
}

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
    config.jpeg_quality = 10;              
    config.fb_count = 2;                   
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
}

void loop() {
  server.handleClient();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    WiFi.reconnect();
    Serial.print("Reconnecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("\nReconnected to WiFi!");
  }
}