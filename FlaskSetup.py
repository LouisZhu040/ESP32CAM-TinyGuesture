from flask import Flask, request
import os
import time

# Create a Flask application instance
app = Flask(__name__)
print("Flask app initialized.")

# Define the directory to store uploaded photos (category: "ye")
pathOfYe = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\Photos\ye"

# Handle photo upload requests via POST
@app.route('/upload', methods=['POST'])
def get_photos():
    # Because the ESP32-CAM sends the photo as binary data,
    # we need to read the data from the request.
    photo = request.get_data()
    print("Received a photo upload request.")
    
    # Set a filename and format for the uploaded photo
    filename = f"ye_{int(time.time())}.jpg"
    print(f"Saving photo as {filename} in {pathOfYe}...")
    
    # Write the received photo data to a file
    # ".save" should be replaced with ".write" to save the binary data
    # And "with" can be used to ensure the file is properly closed after writing
    with open(os.path.join(pathOfYe, filename), 'wb') as f:      
        f.write(photo)
        print(f"Photo saved to {os.path.join(pathOfYe, filename)}")
    
    return f"Photo uploaded as {filename}", 200

# Run the Flask app, let's go!
if __name__ == '__main__':
    # host="0.0.0" allows the app to be accessible from any IP address
    # port=5000 (flask server) is the default port for Flask applications
    app.run(host="0.0.0.0", port=5000, debug=True)

