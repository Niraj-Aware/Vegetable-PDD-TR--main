import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load pre-trained model
model = tf.keras.models.load_model('tomato.h5')

# Define labels for prediction output
labels = ['diseased', 'healthy']

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image
    image = image.resize((150,150))
    # Convert image to numpy array
    image = np.array(image)
    # Scale pixel values to range [0, 1]
    image = image / 255.0
    # Expand dimensions to create batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define function to make prediction on input image
def predict(image):
    # Preprocess input image
    image = preprocess_image(image)
    # Make prediction using pre-trained model
    prediction = model.predict(image)
    # Convert prediction from probabilities to label
    label = labels[np.argmax(prediction)]
    # Return label and confidence score
    return label, prediction[0][np.argmax(prediction)]

# Define Streamlit app
def main():
    # Set app title
    st.title('Tomato Plant Disease Detection')
    # Set app description
    st.write('This app helps you to detect whether a tomato plant is healthy or diseased.')
    st.write('NOTE: This model only works on tomato plant images.')
    # Add file uploader for input image
    tomato_image = st.file_uploader("Choose an image...", type="jpg")
    submit = st.button('Predict')
    # On predict button click
    if submit:
        if tomato_image is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(tomato_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # Displaying the image
            st.image(opencv_image, channels="BGR")
            st.write(opencv_image.shape)
            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (150, 150))
            # Convert image to 4 Dimension
            opencv_image = np.expand_dims(opencv_image, axis=0)
            # Make Prediction
            label, score = predict(Image.fromarray(cv2.cvtColor(opencv_image[0], cv2.COLOR_BGR2RGB)))
            st.write('Prediction: {} (confidence score: {:.2%})'.format(label, score))
            if label in labels:
                if label == 'diseased':
                    st.write('Your tomato plant appears to be diseased. It is recommended to take necessary actions such as removing infected plants, applying fungicides, and maintaining proper plant care to prevent further spread.')
                else:
                    st.write('Your tomato plant appears to be healthy. Keep up with good gardening practices, such as proper watering, fertilizing, and pest control, to maintain its health.')
            else:
                st.write('The uploaded image is not appropriate for this app.')

# Run Streamlit app
if __name__ == '__main__':
    main()
