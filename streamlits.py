import streamlit as st
import requests
import numpy as np
from PIL import Image

st.set_page_config(page_title="fashion MNIST", page_icon = "🧥👜🥾", layout="centered")
st.header("🧥👜🥾 Fashion Mnist")

uploaded_image = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Predicting..."):
            image = Image.open(uploaded_image).convert("L")
            image = image.resize((28, 28))
            image_array = np.array(image).reshape(1, 28, 28, 1) / 255
            st.write("Image shape before sending:", image_array.shape)  # Debug check

            # ✅ Correct JSON structure
            data = {"instances": image_array.tolist()}

            response = requests.post("http://127.0.0.1:8000/prediction", json=data)
            if response.status_code == 200:
                predictions = response.json()
                predictions = predictions.get("predictions", [])
                st.write("Predictions received:", predictions)  # Debug check
                class_names = [
                    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                ]
                st.success(f"Predicted Class: {class_names[predictions]}")
            else:
                st.error("Error in prediction request.")
    else:
        st.warning("Please upload an image before predicting.")