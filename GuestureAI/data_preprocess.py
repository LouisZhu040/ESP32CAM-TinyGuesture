import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Save the path of floders
path_ye = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\Photos\ye"
path_noye = r"E:\ESP32-CAM\Python-edgeAI-GuestureAI\Photos\noye"

# List all photos in folder(ye and noye)
filesInYe = os.listdir(path_ye)
filesInNoye = os.listdir(path_noye)

# Check the photos number
print("The number of photos in Ye''s folder is %d" %(len(filesInYe)))
print("The number of photos in Noye''s folder is %d" %(len(filesInNoye)))

# Set a array to save the photos
photos_ye = []
photos_noye = []

# Set a function of normalizing the images
def normalize_images(images):
    return images / 255.0  # Normalize pixel values to [0, 1]


# Loop through all photos in Ye's folder
for photo in filesInYe:
    # Read the photo
    img = cv2.imread(os.path.join(path_ye, photo))
    # resize all photos
    img = cv2.resize(img, (160, 120))
    # nomoralize the image
    img  = normalize_images(img)
    # Convert the photo to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Because CNN requires a 4D array, we need to reshape the image into a 3D array at first
    gray = gray.reshape(120, 160, 1) # 120 is the height, 160 is the width, and 1 is the channel

    # Temporarily save the photo(3D array) to the array
    photos_ye.append(gray)

# Construct a numpy array from the list of photos, which is a 4D array
photos_ye = np.array(photos_ye)

# Loop through all photos in Noye's folder
for photo in filesInNoye:
    # Read the photo
    img = cv2.imread(os.path.join(path_noye, photo))
    # resize all photos
    img = cv2.resize(img, (160, 120))
    # nomoralize the image
    img  = normalize_images(img)

    # Convert the photo to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Because CNN requires a 4D array, we need to reshape the image
    gray = gray.reshape(120, 160, 1)
    # Temporarily save the photo to the array
    photos_noye.append(gray)

# Construct a numpy array from the list of photos, which is a 4D array
photos_noye = np.array(photos_noye)

# Because we can only 0 and 1 to represent the two classes, we need to create a label array
labels_ye = np.ones((len(photos_ye), 1))  # Label for Ye's photos is 1
labels_noye = np.zeros((len(photos_noye), 1))  # Label for Noye's photos is 0

# Combine the photos and labels into a single dataset
# Combine the array in an x-axis
X = np.concatenate([photos_noye, photos_ye], axis=0)
# Mark the type of photo via "0" or "1"
Y = np.concatenate([labels_ye,labels_noye],axis=0)

# Split the training photos and Vertificational Photos, using a random seed:"42" 
# "shuffle=True" means the array will be randomly distributed before split up.  
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)


