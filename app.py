import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Handwritten Digit Classifier",
    layout="centered"
)

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 AI-Powered Handwritten Digit Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Draw any digit (0-9) below and let our CNN model identify it in real time!</p>", unsafe_allow_html=True)

# --- Canvas ---
st.markdown("## ✏️ Draw Your Digit")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# --- Load the Model ---
model = tf.keras.models.load_model("mnist_cnn_model.keras")

# --- Prediction Logic ---
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Convert to grayscale and resize
    img = Image.fromarray((255 - img[:, :, 0]).astype(np.uint8)).resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    if st.button("🔍 Predict"):
        prediction = np.argmax(model.predict(img_array), axis=1)[0]
        st.success(f"✅ Predicted Digit: **{prediction}**")
else:
    st.info("🖌️ Draw something above to classify.")

# --- Footer ---
st.markdown("""<hr style="margin-top: 50px; margin-bottom: 10px;">""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 16px;">
    Made by <strong>Saee Surve</strong><br>
    <a href="https://github.com/Saee-Surve" target="_blank">GitHub Profile</a>
</div>
""", unsafe_allow_html=True)
