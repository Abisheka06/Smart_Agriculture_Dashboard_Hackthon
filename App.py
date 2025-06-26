import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Agriculture App", layout="wide")

# Sidebar navigation and credits
st.sidebar.title("Smart Agriculture AI")
st.sidebar.info("Developed for the Kapil ItShub Agritech Hackathon")
st.sidebar.markdown("**👨‍💻 Created by:**\n- Abisheka R S\n- Gajapathi E")
st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Navigation", ["🏠 Home", "🌿 Plant Disease Detection", "🌱 Weed Detection"])

# Load models
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

@st.cache_resource
def load_weed_model():
    return tf.keras.models.load_model("weed_detection_model.h5")

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

# ---------------- Home Page ----------------
if page == "🏠 Home":
    st.title(" Smart Agriculture Dashboard")
    st.markdown("""
    Welcome to the Smart Agriculture Dashboard developed for the **Kapil ItShub Agritech Hackathon**.

    This project leverages **deep learning** to:
    -  Detect **plant diseases** using the PlantVillage dataset
    -  Identify **invasive weed species** from field imagery

    ###  Why This Matters
    Farmers face challenges such as:
    - Late detection of diseases
    - Spread of aggressive weeds
    - Time-consuming manual inspection

    Our AI-based tool empowers:
    - 🧑 **Farmers** to make fast, informed decisions
    - 🌍 **Agritech researchers** to develop scalable field solutions

    👉 Use the sidebar to explore:
    - **Plant Disease Detection** using trained CNNs
    - **Weed Detection** based on real field weed imagery

    ### 📦 Tools & Tech Stack:
    - TensorFlow, Keras, Streamlit, PIL, Seaborn
    - Models: MobileNetV2, Custom CNNs
    - Dataset Sources: PlantVillage, DeepWeeds
    
    ### 👨‍💻 Team
    Developed by:
    - **Abisheka R S**  
    - **Gajapathi E**
    """)

# ---------------- Plant Disease Detection Page ----------------
elif page == "🌿 Plant Disease Detection":
    model = load_disease_model()
    st.title("🌿 Plant Disease Detection")
    st.markdown("""
    Upload a leaf image from common crop plants to detect diseases using a CNN trained on 38+ classes from the PlantVillage dataset.
    """)

    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        image_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = plant_class_labels[np.argmax(predictions)]

        st.success(f"Predicted Class: **{predicted_class}**")
        st.subheader("📊 Top 3 Probabilities")
        top_k = 3
        top_k_indices = np.argsort(predictions[0])[::-1][:top_k]
        for i in top_k_indices:
            st.write(f"{plant_class_labels[i]}: {predictions[0][i]*100:.2f}%")
        st.bar_chart(predictions[0])

    if st.checkbox("📈 Show Model Evaluation Metrics"):
        try:
            val_data = tf.keras.preprocessing.image_dataset_from_directory(
                "val",
                image_size=(224, 224),
                batch_size=32,
                label_mode='categorical',
                shuffle=False
            )

            y_true, y_pred = [], []
            for images, labels in val_data:
                preds = model.predict(images)
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(np.argmax(labels.numpy(), axis=1))

            st.dataframe(classification_report(y_true, y_pred, target_names=plant_class_labels, output_dict=True))

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=False, cmap="YlGnBu", xticklabels=plant_class_labels, yticklabels=plant_class_labels)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error loading validation data: {e}")

# ---------------- Weed Detection Page ----------------
elif page == "🌱 Weed Detection":
    model = load_weed_model()
    st.title("🌱 Weed Species Detection")
    st.markdown("""
    Upload a weed image captured in the field to identify the species. The model supports detection of 9 categories including:
    -  Chinee apple
    -  Snake weed
    -  Parthenium
    -  Lantana

    This solution assists farmers and agronomists in **early detection of harmful invasive species**, enabling targeted removal and **precision herbicide use**.
    """)

    uploaded_file = st.file_uploader("📤 Upload a weed image", type=["jpg", "jpeg", "png"], key="weed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        image_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = weed_class_labels[np.argmax(predictions)]

        st.success(f"Predicted Weed Type: **{predicted_class}**")
        st.subheader("📊 Prediction Probabilities")
        top_k = 3
        top_k_indices = np.argsort(predictions[0])[::-1][:top_k]
        for i in top_k_indices:
            st.write(f"{weed_class_labels[i]}: {predictions[0][i]*100:.2f}%")
        st.bar_chart(predictions[0])