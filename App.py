# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model (Replace './tomato_trained_models/1' with the path to your tomato disease model)
model = tf.keras.models.load_model('./tomato.h5')

# Setting Title of App
st.title("Tomato Plant Disease Detection")
st.markdown("Upload an image of the tomato plant leaf")

# Uploading the tomato plant leaf image
plant_image = st.file_uploader("Choose an image...", type="jpg")

# On predict button click
if st.button('Predict'):
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4 Dimension
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        predictions = model.predict(opencv_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = model.classes_[predicted_class_index]  # Assuming your model has classes_ attribute
        st.title(f"This is a tomato leaf with {predicted_class.replace('_', ' ')}")
