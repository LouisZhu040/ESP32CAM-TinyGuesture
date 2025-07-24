# tflite_to_h.py
model_path = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\Deployment\model.tflite"
# Read the TFLite model file
with open(model_path, "rb") as f:
    content = f.read()

# Set up blank array and array name for resorting binary data
lines = []
array_name = "model_tflite"

# Convert the binary content to a C-style array
lines.append(f"unsigned char {array_name}[] = {{")

# Loop through the content and format it
for i, b in enumerate(content):
    # Add a new line every 12 bytes for better readability
    if i % 12 == 0:
        lines.append("\n  ")
    # Convert byte to hex and append it to the current line
    lines[-1] += f"0x{b:02x}, "

lines.append("\n};")
# Add the length of the array, which is the size of the model
lines.append(f"unsigned int {array_name}_len = {len(content)};")

# Create the header file
with open("model.h", "w") as f:
    f.write("\n".join(lines))

print("model.h Created Successfully!")
