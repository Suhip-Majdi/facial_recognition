import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# this code have alot of error

import zipfile

from Algorithem_AI.pridictive import y, X


def unzip_file(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Files extracted successfully to {extract_to}")
    except zipfile.BadZipFile:
        print("Error: The file is not a zip file or it is corrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
zip_path = '/content/AT&T.zip'
extract_to = '/content/Dataset'

unzip_file(zip_path, extract_to)

import cv2 as cv

images = []
labels = []
root = "/content/Dataset"

classes = os.listdir(root)

lab = -1
for cls in classes:
  path2class = os.path.join(root,cls)

  lab += 1
  for img in os.listdir(path2class):
    labels.append(lab)
    path2image = os.path.join(path2class,img)
    image = cv.imread(path2image)
    images.append(image)


images = np.array(images)

## Feature extraction
# For simplicity let us use LBP to extract features fromt he images

def get_lbp_value(center, pixels):
    binary_string = ''.join(['1' if pixel >= center else '0' for pixel in pixels])
    return int(binary_string, 2)

def lbp_image(image):
    if len(image.shape)>2:
      image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    height, width = image.shape
    lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)

    # Iterate over each pixel in the image excluding the border
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = image[i, j]
            # Clockwise order starting from the top-left pixel
            pixels = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],  # P0, P1, P2
                image[i, j+1], image[i+1, j+1], image[i+1, j],   # P3, P4, P5
                image[i+1, j-1], image[i, j-1]                   # P6, P7
            ]
            lbp_image[i-1, j-1] = get_lbp_value(center, pixels)

    return lbp_image

def lbp_feature_vector(lbp_img):

    lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=np.arange(0, 256), range=(0, 256), density=True)

    lbp_hist /= np.sum(lbp_hist)

    return lbp_hist  # Feature vector

# Extract features from images

Features = []
for img in images:

  featurevector = lbp_feature_vector(lbp_image(img))
  Features.append(featurevector)

Features = np.array(Features)
# feature extraction using CNN

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


base_model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


Features = []
for img in images:

  img= cv.resize(img,(224,224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  # Extract features
  features = model.predict(x)
  Features.append(features)



Features = np.array(Features)
Features = np.squeeze(Features, axis=1)

# Split to training and testing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

mdl = LogisticRegression(multi_class='ovr', max_iter=200)
mdl = SVC()
mdl.fit(X_train, y_train)
# Evaluation
from sklearn.metrics import confusion_matrix, classification_report


predictions = mdl.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
class_report = classification_report(y_test, predictions, target_names=classes)
print(class_report)


