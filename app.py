import streamlit as st
import os
import torch
from ultralytics import YOLO
from PIL import Image
import av

# -----------------------------------------------------------------------------
# 1. PATH SETUP (Use Streamlit Secrets)
# -----------------------------------------------------------------------------

# **Replace with the actual path to your trained model!**
trained_model_path = st.secrets["MODEL_PATH"]

# -----------------------------------------------------------------------------
# 2. LOAD MODEL
# -----------------------------------------------------------------------------

try:
    model = YOLO(trained_model_path)  # Load the trained model
    print("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Check GPU availability and set device accordingly
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU is available. Running on GPU.")
else:
    device = 'cpu'
    print("GPU not available. Running on CPU.")

# -----------------------------------------------------------------------------
# 3. UI COMPONENTS
# -----------------------------------------------------------------------------
st.title("Space Station Object Detection")
st.write("Upload an image or take a picture to detect Toolboxes, Oxygen Tanks, and Fire Extinguishers")

# Image Input Options
image_source = st.radio("Select Input Source:", ("Upload Image", "Take a Picture"))

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif image_source == "Take a Picture":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        st.image(image, caption="Camera Input", use_container_width=True)

# Predict Button
if st.button("Detect Objects"):
    if model is not None and 'image' in locals():
        # Run prediction
        results = model(image, verbose=False, device = device)

        # Display Results
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                st.write("No objects detected.")
            else:
                st.write("Objects detected:")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xywhn = box.xywhn[0].tolist()
                    x_center, y_center, width, height = xywhn

                    st.write(f"  - Object: {class_id}, Confidence: {confidence:.2f}, Box: {x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}")

            # Visualize Bounding Boxes
            annotated_image = results[0].plot()  #Visualize
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)

        else:
            st.write("Model not loaded. Check the file paths and try again.")

# -----------------------------------------------------------------------------
# 4. INSTRUCTIONS
# -----------------------------------------------------------------------------
st.sidebar.header("Instructions")
st.sidebar.write("1. Select input source: Upload an image or Take a Picture")
st.sidebar.write("2. Upload an image (JPG, JPEG, or PNG) or take a picture")
st.sidebar.write("3. Click the 'Detect Objects' button.")
st.sidebar.write("4. View the results below the image.")
st.sidebar.write("5. The image with bounding boxes will show")
