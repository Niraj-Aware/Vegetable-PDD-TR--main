import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

# Load models and class names
MODEL = tf.keras.models.load_model('tomato.h5')
TOMATO_MODEL = tf.keras.models.load_model('./tomato_trained_models/1')
PEPPER_MODEL = tf.keras.models.load_model('./pepper_trained_models/1')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
tomato_classes = ['Tomato_healthy', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato_Septoria_leaf_spot',
 'Tomato__Tomato_mosaic_virus', 'Tomato_Leaf_Mold', 'Tomato_Bacterial_spot', 'Tomato_Late_blight',
 'Tomato_Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus']
pepper_classes = ['pepper_healthy', 'pepper_bell_bacterial_spot']

# Set Streamlit page config
st.set_page_config(
    layout="wide",
    page_title='Plant Disease Detection',
)
st.title("Plant Disease Detection & Treatment Recommendation System")
st.write("This application detects diseases in three plants: potato, tomato, and pepper")
options = ["Select One Plant", "Tomato", "Potato", "Pepper"]

# Create a selectbox for the user to choose one option
selected_option = st.selectbox("Select Plant:", options)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def read_file_as_image(data) -> np.array:
    image = np.array(data)
    image = cv2.resize(image, (256, 256))
    return image

def detect_disease(model, class_names):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
        image = read_file_as_image(image)
        image_batch = np.expand_dims(image, axis=0)
        predictions = model.predict(image_batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Disease-specific recommendations
        if predicted_class in ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']:
            st.write("Disease Detected:", predicted_class)
            st.write("Recommendations:")
            # Add recommendations for Potato diseases
            if predicted_class == 'Potato___Early_blight':
                st.write("Recommendations for Potato Early Blight")
            elif predicted_class == 'Potato___Late_blight':
                st.write("Recommendations for Potato Late Blight")
            elif predicted_class == 'Potato___healthy':
                st.write("Plant is Healthy (General Maintenance)")
        elif predicted_class in ['Tomato_healthy', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato_Septoria_leaf_spot',
                                 'Tomato__Tomato_mosaic_virus', 'Tomato_Leaf_Mold', 'Tomato_Bacterial_spot', 'Tomato_Late_blight',
                                 'Tomato_Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus']:
            st.write("Disease Detected:", predicted_class)
            st.write("Recommendations:")
            # Add recommendations for Tomato diseases
            if predicted_class == 'Tomato_healthy':
                st.write("Recommendations for Tomato Healthy")
            elif predicted_class == 'Tomato_Spider_mites_Two_spotted_spider_mite':
                st.write("Recommendations for Tomato Spider Mites")
            elif predicted_class == 'Tomato__Target_Spot':
                st.write("Recommendations for Tomato Target Spot")
            # Add more recommendations for other Tomato diseases
        elif predicted_class in ['pepper_healthy', 'pepper_bell_bacterial_spot']:
            st.write("Disease Detected:", predicted_class)
            st.write("Recommendations:")
            # Add recommendations for Pepper diseases
            if predicted_class == 'pepper_healthy':
                st.write("Recommendations for Pepper Healthy")
            elif predicted_class == 'pepper_bell_bacterial_spot':
                st.write("Recommendations for Pepper Bell Bacterial Spot")
            # Add more recommendations for other Pepper diseases
        else:
            st.write("Unknown disease detected")
        
        st.write("Predicted Class : ", predicted_class, " Confidence Level : ", confidence)

#Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

#On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        # Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        
        if selected_option == 'Potato':
            detect_disease(MODEL, class_names)
        elif selected_option == 'Tomato':
            detect_disease(TOMATO_MODEL, tomato_classes)
        elif selected_option == 'Pepper':
            detect_disease(PEPPER_MODEL, pepper_classes)
        else:
            st.write("Plant not available")
