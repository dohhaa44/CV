import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the fine-tuned model
model = tf.keras.models.load_model("fine_tuned_model.keras")

# set the image size
IMG_SIZE = 224

# class names
class_names = ['CaS','CoS', 'Gum', 'MC','OC','OLP','OT']

# preprocess the image
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))  # change the size to 224x224
    image = np.array(image) / 255.0  # convert to numpy array and normalize
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

# build the streamlit app
st.title("Classification of Oral Lesions Using My Deep Learning Model 🦷 Based on AI")
st.write("upload an image of an oral lesion and I will tell you what it is")

# load the image from the user computer
uploaded_file = st.file_uploader("Choose images....", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Photo you Loaded", use_column_width=True)

    # convert the image to an array and preprocess it
    processed_image = preprocess_image(image)

    # make a prediction using the model
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # display the Result
    st.write(f"Expectation #####: {predicted_class}")
    st.write(f"Confidence ratio**:** {confidence:.2f}%")
