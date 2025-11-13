import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Cat vs Rabbit Classifier 🐱🐇",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS DESIGN ---
st.markdown("""
    <style>
    body {
        background-color: #fafafa;
    }
    .main {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4b9cd3;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .footer {
        text-align: center;
        color: gray;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- PAGE HEADER ---
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🐱🐇 Cat vs Rabbit Image Classifier")
st.write("Upload an image of a **Cat** or **Rabbit**, and this AI model will tell you which one it is!")

st.divider()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_vs_rabbit_model.keras')
    return model

model = load_model()

# --- IMAGE UPLOAD SECTION ---
uploaded_file = st.file_uploader("📤 Upload an image file (JPG or PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess
    img_resized = img.resize((128,128))
    x = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    with st.spinner("Analyzing image... 🧠"):
        pred = model.predict(x)[0][0]
        label = "Rabbit 🐇" if pred > 0.5 else "Cat 🐱"
        confidence = pred if pred > 0.5 else 1 - pred

    st.success(f"### ✅ Prediction: **{label}**")
    st.progress(float(confidence))
    st.caption(f"Confidence: {confidence:.2%}")

else:
    st.info("👆 Upload an image to start prediction!")

st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<p class='footer'>Created by <b>Md Jubayer Islam</b> | Powered by TensorFlow & Streamlit</p>", unsafe_allow_html=True)
