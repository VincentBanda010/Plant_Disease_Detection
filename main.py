import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')

# Prediction function
def model_prediction(image):
    model = load_model()
    img = Image.open(image).convert('RGB')
    img = img.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# Disease class labels 
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Sidebar navigation
st.sidebar.title("🌿 Agronex Dashboard")
app_mode = st.sidebar.selectbox("Navigate", ["🏠 Home", "📄 About", "🔍 Disease Recognition"])

# Home Page
if app_mode == "🏠 Home":
    st.title("🌱 Plant Disease Recognition System")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the **Agronex Plant Disease Recognition System**!  
    Helping farmers protect their crops with the power of AI.

    ### 🚀 How It Works:
    1. Navigate to **Disease Recognition**
    2. Upload an image of a diseased plant
    3. Receive instant diagnosis and suggested actions

    ### 💡 Why Agronex?
    - ✅ Accurate ML Predictions  
    - 🧑‍🌾 Farmer-Friendly Interface  
    - ⚡ Fast and Reliable

    👉 Get started now by choosing **Disease Recognition** from the sidebar!
    """)

# About Page
elif app_mode == "📄 About":
    st.title("📘 About Agronex")
    st.markdown("""
    **Agronex** is an AI-powered platform designed to detect plant diseases from leaf images using deep learning.

    #### 📊 Dataset Summary
    - **87,000+ RGB images** across **38 classes**
    - Augmented dataset for better generalization
    - Train/Validation Split: **80/20**

    #### 📁 Structure
    - `Train/`: 70,295 images  
    - `Valid/`: 17,572 images  
    - `Test/`: 33 images

    Dataset credits: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
    """)

# Prediction Page
elif app_mode == "🔍 Disease Recognition":
    st.title("🧪 Disease Recognition")
    uploaded_file = st.file_uploader("📤 Upload an image of a plant leaf", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("🧠 Predict"):
            with st.spinner("Analyzing the image..."):
                prediction_index = model_prediction(uploaded_file)
                predicted_disease = CLASS_NAMES[prediction_index]
                st.success(f"✅ Prediction: **{predicted_disease.replace('_', ' ')}**")
