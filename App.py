# Import Required Libraries
import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st

# Load plant disease classification model
filename = 'plant_disease_classification_model.pkl'
model = pickle.load(open(filename, 'rb'))

# Load plant disease labels
filename = 'plant_disease_label_transform.pkl'
image_labels = pickle.load(open(filename, 'rb'))

# Dimensions of resized image
DEFAULT_IMAGE_SIZE = (256, 256)

def convert_image_to_array(image):
    image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
    return image

def detect_disease(image_path):
    image = cv2.imread(image_path)
    image_array = convert_image_to_array(image)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image, 0)
    result = model.predict_classes(np_image)
    predicted_class = image_labels.classes_[result][0]
    confidence = model.predict_proba(np_image).max()
    return predicted_class, confidence

# Update the detect_disease function in your Streamlit app
def detect_disease(model, class_names):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        predicted_class, confidence = detect_disease(uploaded_file)
        
        # Disease-specific recommendations
        # Add your recommendations logic here
        
        st.write("Predicted Class : ", predicted_class, " Confidence Level : ", confidence)
        st.write("")
        st.write("To obtain more details and accurate treatment, please visit the nearest pharmaceutical or plant pharma facility")

# Use detect_disease function in your Streamlit app based on the selected option
if selected_option == 'Potato':
    detect_disease(model, class_names)
elif selected_option == 'Tomato':
    detect_disease(model, class_names)
elif selected_option == 'Corn':
    detect_disease(model, class_names)
else:
    st.write("Plant not available")
