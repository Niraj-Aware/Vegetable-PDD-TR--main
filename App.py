# Import necessary libraries
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64
import cv2

# Load models and class names
#MODEL = tf.keras.models.load_model('./potato_trained_models/1/')
MODEL = tf.keras.models.load_model('tomato.h5')
TOMATO_MODEL = tf.keras.models.load_model('./tomato_trained_models/1')
#TOMATO_MODEL = tf.keras.models.load_model('tomato.h5')
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
        if predicted_class=="Potato___Early_blight" in predicted_class:
            st.write("Disease Detected: Potato Early Blight")
            st.write("Recommendations:")
            st.write("1. Fungicide Application: Use fungicides containing active ingredients like chlorothalonil, mancozeb, or copper-based compounds. Follow manufacturer's instructions.")
            st.write("2. Crop Rotation: Avoid planting potatoes in the same area year after year.")
            st.write("3. Proper Spacing: Plant potatoes at recommended spacing for good air circulation.")
            st.write("4. Remove Infected Leaves: Early removal of infected leaves is crucial.")
            st.write("5. Mulching: Apply mulch around the base of plants to prevent soil splashing onto leaves.")
        
        elif predicted_class=="Potato___Late_blight" in predicted_class:
            st.write("Disease Detected: Potato Late Blight")
            st.write("Recommendations:")
            st.write("1. Fungicide Application: Use fungicides containing active ingredients like copper, mancozeb, or systemic fungicides like chlorothalonil.")
            st.write("2. Proper Watering: Avoid overhead watering to reduce leaf wetness.")
            st.write("3. Crop Rotation: Rotate crops to prevent pathogen buildup.")
            st.write("4. Plant Resistant Varieties: Consider resistant potato varieties.")
            st.write("5. Early Detection: Monitor plants for signs of late blight; take action immediately.")
        
        elif predicted_class=="Potato___healthy" in predicted_class:
            st.write("Plant is Healthy (General Maintenance)")
            st.write("Recommendations:")
            st.write("1. Proper Planting: Use certified disease-free potato seed pieces.")
            st.write("2. Crop Rotation: Practice crop rotation to prevent disease buildup in the soil.")
            st.write("3. Fertilization: Apply balanced fertilizers based on soil test results.")
            st.write("4. Weed Control: Keep the area around potatoes weed-free.")
            st.write("5. Monitoring: Regularly inspect plants for signs of disease, pests, or nutrient deficiencies.")
            st.write("6. Harvest and Storage: Handle harvested potatoes gently and store them in cool, dark, well-ventilated conditions.")

        elif predicted_class=="pepper_bell_bacterial_spot" in predicted_class:
            # Recommendations for Pepper Bell Bacterial Spot
            st.write("Disease Detected: Pepper Bell Bacterial Spot")
            st.write("Recommendations:")
            st.write("A. Preventive Measures:")
            st.write("1. Seed Selection: Start with disease-free seeds or transplants from a reputable source.")
            st.write("2. Sanitation: Practice good sanitation in your garden, including cleaning tools and equipment between uses.")
            st.write("3. Crop Rotation: Avoid planting peppers or other susceptible crops in the same area year after year.")
            st.write("B. Cultural Practices:")
            st.write("1. Proper Spacing: Plant peppers at recommended spacing to allow for good air circulation.")
            st.write("2. Drip Irrigation: Use drip irrigation or soaker hoses to water the plants at the base, avoiding wetting the foliage.")
            st.write("3. Mulching: Apply mulch to prevent soil splashing onto leaves, which can spread the bacteria.")
            st.write("4. Pruning: Prune and remove affected leaves and stems promptly to reduce disease spread.")
            st.write("C. Chemical Treatments:")
            st.write("1. Copper-based Fungicides: Copper-based fungicides can help control bacterial spot. Apply these according to label instructions, especially during periods of warm, wet weather.")
            st.write("2. Copper Sprays: Copper sprays can be used as a preventive measure before symptoms appear or as a treatment after detection.")

        elif predicted_class=="pepper_healthy" in predicted_class:
            # Recommendations for Pepper Healthy
            st.write("Plant is Healthy (General Maintenance)")
            st.write("Recommendations:")
            st.write("Proper Planting and Care:")
            st.write("1. Select Disease-resistant Varieties: Choose pepper varieties known to be resistant to common diseases in your area.")
            st.write("2. Good Soil Preparation: Ensure well-draining soil with organic matter and proper pH levels.")
            st.write("3. Fertilization: Apply balanced fertilizers based on soil test results to meet the nutritional needs of pepper plants.")
            st.write("4. Watering: Water consistently, avoiding overwatering.")

        elif predicted_class=="Tomato_healthy" in predicted_class:
            st.write("Disease Detected: Tomato Healthy")
            st.write("Recommendations:")
            st.write("1. Preventive Measures: Start with disease-free seeds or transplants from a reputable source.")
            st.write("2. Crop Rotation: Avoid planting tomatoes or other solanaceous crops in the same area year after year.")
            st.write("3. Proper Spacing: Plant tomatoes at recommended spacing to allow for good air circulation.")
            st.write("4. Mulching: Apply organic mulch to prevent soil splashing onto leaves and reduce weed competition.")
            st.write("5. Watering: Use drip irrigation or soaker hoses to water at the base, avoiding wetting the foliage.")
            st.write("6. Fertilization: Provide proper nutrition based on soil test results.")
        
        elif predicted_class=="Tomato_Spider_mites_Two_spotted_spider_mite" in predicted_class:
            st.write("Disease Detected: Tomato Spider Mites")
            st.write("Recommendations:")
            st.write("1. Cultural Practices: Maintain a clean garden environment and remove weeds, which can harbor mites.")
            st.write("2. Spraying with Water: Spray plants with a strong stream of water to dislodge mites.")
            st.write("3. Miticides: If infestation is severe, consider using miticides as per the label instructions.")
        
        elif predicted_class=="Tomato__Target_Spot" in predicted_class:
            st.write("Disease Detected: Tomato Target Spot")
            st.write("Recommendations:")
            st.write("1. Fungicides: Apply fungicides containing active ingredients like chlorothalonil or copper-based compounds, following label instructions.")
        
        elif predicted_class=="Tomato_Septoria_leaf_spot" in predicted_class:
            st.write("Disease Detected: Tomato Septoria Leaf Spot")
            st.write("Recommendations:")
            st.write("1. Fungicides: Apply fungicides containing active ingredients like chlorothalonil or copper-based compounds, following label instructions.")
            st.write("2. Remove Infected Leaves: Prune and remove infected leaves to reduce disease spread.")
        
        elif predicted_class=="Tomato__Tomato_mosaic_virus" in predicted_class:
            st.write("Disease Detected: Tomato Mosaic Virus")
            st.write("Recommendations:")
            st.write("1. No cure: There is no cure for viral diseases. Prevent infection by controlling aphids, which can transmit the virus, and removing infected plants promptly to prevent spread.")
        
        elif predicted_class=="Tomato_Leaf_Mold" in predicted_class:
            st.write("Disease Detected: Tomato Leaf Mold")
            st.write("Recommendations:")
            st.write("1. Fungicides: Apply fungicides containing active ingredients like chlorothalonil or copper-based compounds, following label instructions.")
            st.write("2. Improve Air Circulation: Prune plants to improve air circulation and reduce humidity.")
        
        elif predicted_class=="Tomato_Bacterial_spot" in predicted_class:
            st.write("Disease Detected: Tomato Bacterial Spot")
            st.write("Recommendations:")
            st.write("1. Copper-based Fungicides: Copper-based fungicides can help control bacterial spot. Apply according to label instructions.")
            st.write("2. Prune Infected Parts: Prune and remove affected leaves and stems promptly.")
        
        elif predicted_class=="Tomato_Late_blight" in predicted_class:
            st.write("Disease Detected: Tomato Late Blight")
            st.write("Recommendations:")
            st.write("1. Fungicides: Apply fungicides containing active ingredients like chlorothalonil or copper-based compounds, following label instructions.")
            st.write("2. Remove Infected Plants: Remove and destroy infected plants to prevent further spread.")
        
        elif predicted_class=="Tomato_Early_blight" in predicted_class:
            st.write("Disease Detected: Tomato Early Blight")
            st.write("Recommendations:")
            st.write("1. Fungicides: Apply fungicides containing active ingredients like chlorothalonil or copper-based compounds, following label instructions.")
            st.write("2. Remove Infected Leaves: Prune and remove infected leaves to reduce disease spread.")
        
        elif predicted_class=="Tomato__Tomato_YellowLeaf__Curl_Virus" in predicted_class:
            st.write("Disease Detected: Tomato Yellow Leaf Curl Virus")
            st.write("Recommendations:")
            st.write("1. Virus Management: Control whiteflies, which transmit the virus, using insecticides or reflective mulch.")
            st.write("2. Resistant Varieties: Plant tomato varieties that are resistant to the virus if available in your region.")
        else:
            st.write("RE-CAPTURE THE VALID PLANT")
        
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
