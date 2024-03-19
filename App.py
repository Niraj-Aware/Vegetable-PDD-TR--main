#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

#Loading the Model
model = load_model('./dataset/disease.h5')

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

#Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        # Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        
        # Displaying disease and recommendations
        st.header(f"Disease detected: {result.split('-')[0]} leaf with {result.split('-')[1]}")
        
        if result == 'Tomato-Bacterial_spot':
            st.subheader("Recommendations:")
            st.write("1. Use disease-free seeds or transplants.")
            st.write("2. Practice crop rotation to avoid soilborne diseases.")
            st.write("3. Use drip irrigation to avoid wetting the foliage.")
            st.write("4. Apply copper-based fungicides.")
            st.write("5. Remove and destroy infected plants promptly.")
        else:
            st.write("No recommendations available for this disease.")
