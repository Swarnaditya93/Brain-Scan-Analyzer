import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.model import MyModel, load_model
from src.utils import predict

# --- Set Page to Wide Mode ---
st.set_page_config(layout="wide")

# --- CSS Function ---
def load_css(file_name):
    """Function to load a local CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join("models", "model_38")
model = load_model(model_path, device)

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- Descriptions for each class ---
TUMOR_DESCRIPTIONS = {
    "No Tumor": "The model has not detected any signs of a glioma, meningioma, or pituitary tumor in this scan. This indicates a healthy scan according to the model's training.",
    "Pituitary": "A Pituitary Tumor is an abnormal growth in the pituitary gland, a small, pea-sized organ at the base of the brain. Most are benign (non-cancerous) and grow slowly.",
    "Glioma": "A Glioma is a common type of tumor that originates in the glial cells (the supportive 'gluey' cells) of the brain or spinal cord. They can be benign or malignant and are graded by how fast they grow.",
    "Meningioma": "A Meningioma is a tumor that forms on the meninges, the protective membranes that cover the brain and spinal cord. Most meningiomas are benign (non-cancerous) and slow-growing.",
    "Other": "The model has detected an abnormality that does not fall into the Glioma, Meningioma, or Pituitary categories. Please consult a medical professional for further analysis."
}

# map labels from int to string
label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}

# process image got from user before passing to the model
def preprocess_image(image):
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

# sample image loader
@st.cache_data
def load_sample_images(sample_images_dir):
    sample_image_files = os.listdir(sample_images_dir)
    sample_images = []
    for sample_image_file in sample_image_files:
        sample_image_path = os.path.join(sample_images_dir, sample_image_file)
        sample_image = Image.open(sample_image_path).convert("RGB")
        # --- THIS IS THE FIX ---
        # The 1200x1200 resize line has been removed.
        # ---
        sample_images.append((sample_image_file, sample_image))
    return sample_images

# --- Streamlit app ---

# Load the CSS
load_css("style.css")

# --- MODIFIED: New Title ---
st.title("AI-Powered Brain Scan Analyzer for Tumor Classification")
st.write("---") # Adds a horizontal line

# Display sample images section
st.subheader("Sample Images")
st.write(
    "Here are some sample images. Your uploaded image should be similar to these for best results."
)

sample_images_dir = "sample"
try:
    sample_images = load_sample_images(sample_images_dir)

    # Create a grid layout for sample images
    num_cols = 3  # Number of columns in the grid
    cols = st.columns(num_cols)

    for i, (sample_image_file, sample_image) in enumerate(sample_images):
        col_idx = i % num_cols
        with cols[col_idx]:
            # --- THIS IS THE FIX ---
            # Changed width=60 to use_column_width=True
            st.image(sample_image, caption=f"Sample {i+1}", use_column_width=True)

except FileNotFoundError:
     st.error("Sample images not found. Make sure the 'sample' folder is in your project directory.")


st.write("---") # Adds a horizontal line

# --- MODIFIED: More compact upload section ---
st.write("**Upload an MRI scan below to get an instant classification.**")

# image from user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing the image..."):
            # Preprocess the image
            preprocessed_image = preprocess_image(image).to(device)
            # Make prediction
            predicted_class = predict(model, preprocessed_image, device)
            
            # Get the prediction name and description
            prediction_name = label_dict[predicted_class]
            prediction_description = TUMOR_DESCRIPTIONS[prediction_name]

        # Display the prediction
        st.write(
            f"<h2 style='text-align: left;'>Prediction: {prediction_name}</h2>",
            unsafe_allow_html=True,
        )
        
        # --- NEW: Display the description ---
        st.subheader("About this Result:")
        if prediction_name == "No Tumor":
            st.success(prediction_description)
        else:
            st.warning(prediction_description)
        
        st.info(
            "**Disclaimer:** This is an AI-generated analysis and not a substitute for "
            "professional medical advice. Please consult a qualified radiologist."
        )