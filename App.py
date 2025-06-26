import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

st.set_page_config(page_title="Smart Agriculture App", layout="wide")

# Sidebar navigation and credits
st.sidebar.title("Smart Agriculture AI")
st.sidebar.info("Developed for the Kapil ItShub Agritech Hackathon")
st.sidebar.markdown("**👨‍💻 Created by:**\n- Abisheka R S\n- Gajapathi E")
st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Navigation", ["🏠 Home", "🌿 Plant Disease Detection", "🌱 Weed Detection"])

# Load ONNX models
@st.cache_resource
def load_plant_model():
    return ort.InferenceSession("plant_disease_model.onnx")

@st.cache_resource
def load_weed_model():
    return ort.InferenceSession("weed_detection_model.onnx")

# Class labels
plant_class_labels = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Healthy', 'Cherry - Powdery Mildew',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Healthy', 'Corn - Northern Leaf Blight',
    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Healthy', 'Grape - Leaf Blight',
    'Orange - Huanglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper - Bell - Bacterial Spot', 'Pepper - Bell - Healthy',
    'Potato - Early Blight', 'Potato - Healthy', 'Potato - Late Blight',
    'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Healthy', 'Strawberry - Leaf Scorch',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Healthy', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites - Two-Spotted Spider Mite',
    'Tomato - Target Spot', 'Tomato - Mosaic Virus', 'Tomato - Yellow Leaf Curl Virus'
]

weed_class_labels = [
    'Chinee apple', 'Lantana', 'Negative', 'Parkinsonia', 'Parthenium',
    'Prickly acacia', 'Rubber vine', 'Siam weed', 'Snake weed'
]

# Prediction helper
def predict(session, image_array):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    return outputs[0]

# ---------------- Home Page ----------------
if page == "🏠 Home":
    st.title(" Smart Agriculture Dashboard")
    st.markdown("""
    Welcome to the Smart Agriculture Dashboard developed for the **Kapil ItShub Agritech Hackathon**.

    This project leverages **deep learning** to:
    - Detect **plant diseases** using the PlantVillage dataset
    - Identify **invasive weed species** from field imagery

    ###  Why This Matters
    Farmers face challenges such as:
    - Late detection of diseases
    - Spread of aggressive weeds
    - Time-consuming manual inspection

    Our AI-based tool empowers:
    - 🧑 Farmers to make fast, informed decisions
    - 🌍 Agritech researchers to develop scalable field solutions

    ### 📦 Tools & Tech Stack:
    - ONNX, Streamlit, PIL, NumPy
    - Models: MobileNetV2, Custom CNNs
    - Dataset Sources: PlantVillage, DeepWeeds
    """)

# ---------------- Plant Disease Detection Page ----------------
elif page == "🌿 Plant Disease Detection":
    model = load_plant_model()
    st.title("🌿 Plant Disease Detection")
    st.markdown("Upload a leaf image to detect diseases (trained on 38 classes).")

    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        img_array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
        predictions = predict(model, img_array)
        pred_idx = np.argmax(predictions)
        st.success(f"Predicted Class: **{plant_class_labels[pred_idx]}**")

        st.subheader("📊 Top 3 Probabilities")
        top_k = predictions[0].argsort()[-3:][::-1]
        for idx in top_k:
            st.write(f"{plant_class_labels[idx]}: {predictions[0][idx]*100:.2f}%")
        st.bar_chart(predictions[0])

# ---------------- Weed Detection Page ----------------
elif page == "🌱 Weed Detection":
    model = load_weed_model()
    st.title("🌱 Weed Species Detection")
    st.markdown("Upload a weed image to detect the species (9 classes).")

    uploaded_file = st.file_uploader("📤 Upload a weed image", type=["jpg", "jpeg", "png"], key="weed")
    if uploaded_file:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        img_array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)
        predictions = predict(model, img_array)
        pred_idx = np.argmax(predictions)
        st.success(f"Predicted Weed Type: **{weed_class_labels[pred_idx]}**")

        st.subheader("📊 Prediction Probabilities")
        top_k = predictions[0].argsort()[-3:][::-1]
        for idx in top_k:
            st.write(f"{weed_class_labels[idx]}: {predictions[0][idx]*100:.2f}%")
        st.bar_chart(predictions[0])
