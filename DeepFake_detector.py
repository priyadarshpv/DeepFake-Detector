import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
import cv2
import tempfile

# Load the saved model
model = load_model("xception_deepfake_detection_model.h5")

# Set the title and description
st.title("DEEPFAKE DETECTION APP 🔍")
st.markdown("""
Upload an image to check if it's **REAL** or **FAKE**!
""")

# Add a file uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Define preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Match the input size of your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize (same as during training)
    return img_array

def process_video(uploaded_file):
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    st.write(f"Total Frames: {frame_count}, FPS: {fps}")

    # Process each frame
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = PIL.Image.fromarray(frame)
        processed_img = preprocess_image(img)

        # Predict
        prediction = model.predict(processed_img)
        confidence = prediction[0][0]

        # Display results for each frame
        st.image(frame, caption=f"Frame {i + 1}", width=300)
        if confidence > 0.5:
            st.success(f"**Real** (Confidence: {confidence * 100:.2f}%)")
        else:
            st.error(f"**Fake** (Confidence: {(1 - confidence) * 100:.2f}%)")

    # Release the video capture object
    cap.release()
    
# Run inference when an image is uploaded
if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
    st.write(file_details)

    if uploaded_file.type.startswith("image"):
        # Handle image
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        confidence = prediction[0][0]
        if confidence > 0.5:
            st.success(f"**Real** (Confidence: {confidence * 100:.2f}%")
        else:
            st.error(f"**Fake** (Confidence: {(1 - confidence) * 100:.2f}%")

    elif uploaded_file.type.startswith("video"):
        # Handle video
        st.video(uploaded_file)
        process_video(uploaded_file)
        
    # Optional: Add a warning/note
    st.warning("⚠️ Note: This is a demo, Accuracy depends on model")
