import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Load the saved model
model = load_model(r"C:\Users\DELL\Downloads\xception_deepfake_detection_model.h5")

# Set the title and description
st.title("DEEPFAKE DETECTION APP 🔍")
st.markdown("""
Upload an image to check if it's **REAL** or **FAKE**!
""")

# Add a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Match the input size of your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize (same as during training)
    return img_array

# Run inference when an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=False,width=140)

    # Preprocess and predict
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    confidence = prediction[0][0]

    # Display result
    if confidence > 0.5:
        st.success(f"**REAL** (Confidence: {confidence * 100:.2f}%)")  # Real = 1
    else:
        st.error(f"**FAKE** (Confidence: {(1 - confidence) * 100:.2f}%)")  # Fake = 0

    # Optional: Add a warning/note
    st.warning("⚠️ Note: This is a demo, Accuracy depends on model")