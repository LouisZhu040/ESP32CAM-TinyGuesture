// include necessary libraries
#include <Arduino.h>
#include <camera_pins.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <HTTPClient.h>
// Define camera pins for AI Thinker model
#define CAMERA_MODEL_AI_THINKER

// Define the GPIO pins for the LED
const int LED_PIN = 4;
const char* ssid = "iPhone 16 Pro Max";   // Please replace with your WiFi SSID if you want to use WiFi
const char* password = "20070121Louis";   // Please replace with your WiFi password if you want to use WiFi

// Set fb to null pointer(Global variable)
camera_fb_t *fb = NULL;

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

      // Use the LED to indicate the upload failed
      digitalWrite(LED_PIN, HIGH); // Turn on the LED to indicate the upload failed
      delay(200); // Keep the LED on for 0.2 second
      digitalWrite(LED_PIN, LOW); // Turn off the LED
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
  savePhoto();
  releasePhoto();
  delay(1000); // Wait for a second before the next capture
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
    config.jpeg_quality = 40;
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

  // Set the mode of the LED pin to OUTPUT
  pinMode(LED_PIN, OUTPUT);
  // Set the LED pin to LOW initially
  digitalWrite(LED_PIN, LOW);
}

void loop() {
  handlePhoto(); // Call the function to handle photo capture and submission

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