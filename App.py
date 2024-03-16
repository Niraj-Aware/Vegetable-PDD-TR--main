import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load models and class names
MODEL = tf.keras.models.load_model('./potato_trained_models/1/')
TOMATO_MODEL = tf.keras.models.load_model('./tomato_trained_models/1')
PEPPER_MODEL = tf.keras.models.load_model('./pepper_trained_models/1')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
tomato_classes = ['Tomato_healthy', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                  'Tomato_Septoria_leaf_spot', 'Tomato__Tomato_mosaic_virus', 'Tomato_Leaf_Mold',
                  'Tomato_Bacterial_spot', 'Tomato_Late_blight', 'Tomato_Early_blight',
                  'Tomato__Tomato_YellowLeaf__Curl_Virus']
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
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(predictions[0])
        
        # Disease-specific recommendations
        if "Potato" in selected_option:
            if "Early_blight" in predicted_class:
                st.write("Disease Detected: Potato Early Blight")
                st.write("Recommendations:")
                st.write("1. Fungicide Application: Use fungicides containing active ingredients like chlorothalonil, mancozeb, or copper-based compounds. Follow manufacturer's instructions.")
                st.write("2. Crop Rotation: Avoid planting potatoes in the same area year after year.")
                st.write("3. Proper Spacing: Plant potatoes at recommended spacing for good air circulation.")
                st.write("4. Remove Infected Leaves: Early removal of infected leaves is crucial.")
                st.write("5. Mulching: Apply mulch around the base of plants to prevent soil splashing onto leaves.")
            elif "Late_blight" in predicted_class:
                st.write("Disease Detected: Potato Late Blight")
                st.write("Recommendations:")
                st.write("1. Fungicide Application: Use fungicides containing active ingredients like copper, mancozeb, or systemic fungicides like chlorothalonil.")
                st.write("2. Proper Watering: Avoid overhead watering to reduce leaf wetness.")
                st.write("3. Crop Rotation: Rotate crops to prevent pathogen buildup.")
                st.write("4. Plant Resistant Varieties: Consider resistant potato varieties.")
                st.write("5. Early Detection: Monitor plants for signs of late blight; take action immediately.")
            elif "healthy" in predicted_class:
                st.write("Plant is Healthy (General Maintenance)")
                st.write("Recommendations:")
                st.write("1. Proper Planting: Use certified disease-free potato seed pieces.")
                st.write("2. Crop Rotation: Practice crop rotation to prevent disease buildup in the soil.")
                st.write("3. Fertilization: Apply balanced fertilizers based on soil test results.")
                st.write("4. Weed Control: Keep the area around potatoes weed-free.")
                st.write("5. Monitoring: Regularly inspect plants for signs of disease, pests, or nutrient deficiencies.")
                st.write("6. Harvest and Storage: Handle harvested potatoes gently and store them in cool, dark, well-ventilated conditions.")
        
        elif "Tomato" in selected_option:
            st.write("Not implemented yet for Tomato.")
            # You can implement similar logic for tomato diseases
        
        elif "Pepper" in selected_option:
            st.write("Not implemented yet for Pepper.")
            # You can implement similar logic for pepper diseases
        
        st.write("Predicted Class : ", predicted_class, " Confidence Level : ", confidence)
        st.write("")
        st.write("To obtain more details and accurate treatment, please visit the nearest pharmaceutical or plant pharma facility")

if selected_option == 'Potato':
    detect_disease(MODEL, class_names)
elif selected_option == 'Tomato':
    detect_disease(TOMATO_MODEL, tomato_classes)
elif selected_option == 'Pepper':
    detect_disease(PEPPER_MODEL, pepper_classes)
else:
    st.write("Plant not available")

