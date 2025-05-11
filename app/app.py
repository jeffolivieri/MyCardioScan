# app/app.py

import streamlit as st
import torch
import numpy as np
import nibabel as nib
import cv2
import time
import tempfile
import os

from src.model import AorticDissectionClassifier
from src.explainable_ai import generate_gradcam
from pytorch_grad_cam.utils.image import show_cam_on_image

# ------------------------
# Elegant Medical Styling
# ------------------------

st.set_page_config(page_title="My Cardio Scan", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Open Sans', 'Roboto', sans-serif;
        background-color: #f6f9fc;
        color: #333333;
    }

    .stButton>button {
        font-weight: 600;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }

    .stProgress>div>div>div>div {
        background-color: #007bff;
    }

    .stSidebar {
        background-color: #eef2f7;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Utility Functions
# ------------------------

def load_nifti_from_file(uploaded_file):
    """Save uploaded file temporarily and load it as NIfTI."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    img = nib.load(tmp_path)
    data = img.get_fdata()
    return np.array(data)

def normalize(volume):
    volume = np.clip(volume, 0, 512)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-5)
    return volume

def preprocess_volume(volume, target_shape=(64, 128, 128)):
    volume = normalize(volume)
    depth_factor = target_shape[0] / volume.shape[-1]
    height_factor = target_shape[1] / volume.shape[0]
    width_factor = target_shape[2] / volume.shape[1]

    resized = np.zeros(target_shape, dtype=np.float32)

    for i in range(target_shape[0]):
        idx = int(i / depth_factor)
        idx = np.clip(idx, 0, volume.shape[-1] - 1)
        slice_img = volume[:, :, idx]
        slice_resized = cv2.resize(slice_img, (target_shape[2], target_shape[1]))
        resized[i] = slice_resized

    return resized

# ------------------------
# Streamlit App
# ------------------------

st.title("Rivetti Cardio Scanner")
st.caption("Em processo de treinamento pelo Prof. Dr. Luiz Antonio Rivetti")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", [
    "Upload and Predict",
    "3D Volume Visualizer",
    "Explainable AI 🔍",
    "📚 Instructions"   # 👈 New page!
])


# Upload file
uploaded_file = st.file_uploader("Upload a CT Scan (.nii.gz)", type=["nii.gz", "gz"])

if uploaded_file is not None:
    try:
        volume = load_nifti_from_file(uploaded_file)
        volume = preprocess_volume(volume)
        st.success("✅ CT scan uploaded and processed successfully!")
    except Exception as e:
        st.error(f"❌ Error reading uploaded file: {str(e)}")
        volume = None
else:
    volume = None

# ------------------------
# Pages
# ------------------------

if page == "Upload and Predict":
    st.header("AI Prediction")

    if volume is None:
        st.warning("⚠️ Please upload a CT scan first.")
    else:
        if st.button("Run Model Prediction"):
            with st.spinner('Running prediction...'):

                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.04)
                    progress_bar.progress(percent_complete)

                tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                model = AorticDissectionClassifier()
                model.load_state_dict(torch.load("model_best.pth", map_location=torch.device("cpu")))
                model.eval()

                with torch.no_grad():
                    prob = model(tensor).item()

            st.success('✅ Prediction Completed.')

            st.subheader("🧠 Prediction Result")
            st.write(f"**Dissection Probability:** {prob:.2%}")

            if prob > 0.5:
                st.error("⚠️ High Risk of Aortic Dissection Detected!")
            else:
                st.success("✅ No Dissection Detected.")

elif page == "3D Volume Visualizer":
    st.header("🖼️ 3D CT Volume Viewer")

    if volume is None:
        st.warning("⚠️ Please upload a CT scan first.")
    else:
        st.write(f"Volume Shape: `{volume.shape}`")

        axial_idx = st.slider("Axial Slice (Z-axis)", 0, volume.shape[0]-1, volume.shape[0]//2)
        coronal_idx = st.slider("Coronal Slice (Y-axis)", 0, volume.shape[1]-1, volume.shape[1]//2)
        sagittal_idx = st.slider("Sagittal Slice (X-axis)", 0, volume.shape[2]-1, volume.shape[2]//2)

        st.subheader("🧠 Multiplanar Views")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(volume[axial_idx,:,:], caption=f"Axial Slice {axial_idx}", clamp=True, use_container_width=True)

        with col2:
            st.image(volume[:,coronal_idx,:], caption=f"Coronal Slice {coronal_idx}", clamp=True, use_container_width=True)

        with col3:
            st.image(volume[:,:,sagittal_idx], caption=f"Sagittal Slice {sagittal_idx}", clamp=True, use_container_width=True)

elif page == "Explainable AI 🔍":
    st.header("🔍 Explainable AI - GradCAM Heatmap")

    if volume is None:
        st.warning("⚠️ Please upload a CT scan first.")
    else:
        with st.spinner('Generating GradCAM Heatmap...'):

            tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            model = AorticDissectionClassifier()
            model.load_state_dict(torch.load("model_best.pth", map_location=torch.device("cpu")))
            model.eval()

            target_layer = model.conv2

            heatmap = generate_gradcam(model, tensor, target_layer)

            # Prepare slice
            mid_slice_idx = tensor.shape[2] // 2
            slice_img = volume[:, :, mid_slice_idx]

            slice_img_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
            slice_img_rgb = np.repeat(slice_img_norm[..., np.newaxis], 3, axis=-1)

            heatmap_resized = cv2.resize(heatmap, (slice_img_rgb.shape[1], slice_img_rgb.shape[0]))

            cam_overlay = show_cam_on_image(slice_img_rgb, heatmap_resized, use_rgb=True)

            st.image(cam_overlay, caption=f"GradCAM Overlay on Slice {mid_slice_idx}", use_container_width=True)
elif page == "📚 Instructions":
    st.header("📚 Instructions - How to Use MyCardioScan")

    st.markdown("""
    ### 1️⃣ Preprocessing CT Scans
    - Convert your CT scan DICOM files to NIfTI (.nii.gz) format.
    - Use the provided `preprocessing.py` script.
    - Recommended shape: 64 slices × 128 × 128 resolution.
    - Normalize intensity values between 0 and 512.

    ### 2️⃣ Labeling the Scans
    - Use the `label_gui.py` tool (Streamlit app).
    - Label each scan:
      - **1** = Aortic Dissection present
      - **0** = Normal aorta
    - Save labels into a `labels.csv` file (scan_id,label).

    ### 3️⃣ Training the Model
    - Run `train_with_dashboard.py` to train BodyVerse 3D CNN.
    - Settings:
      - Batch size: 4
      - Learning rate: 1e-4
      - Early stopping after 5 epochs without improvement
    - Model checkpoint (`model_best.pth`) will be saved automatically.

    ### 4️⃣ Validation and Testing
    - The training script automatically evaluates on the validation set.
    - You can manually run `evaluate.py` to test performance.

    ### 5️⃣ Running Predictions
    - Upload a NIfTI file using the "Upload and Predict" page.
    - BodyVerse will predict dissection probability.
    - If high probability (> 50%), it flags high risk.

    ### 6️⃣ Explainable AI - GradCAM
    - Navigate to the "Explainable AI 🔍" page.
    - Visualize GradCAM heatmaps highlighting suspicious regions.
    - GradCAM is based on convolutional feature maps of the model.

    ### 📌 Notes
    - For best results, ensure high-quality NIfTI files.
    - Labeling accuracy is crucial for model performance.
    - Pretrained model available (`model_best.pth`) but retraining is recommended for new datasets.
    """, unsafe_allow_html=True)
