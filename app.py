import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils.selective_search import find_regions
from utils.prediction import detect_ships

# Page configuration
st.set_page_config(
    page_title="Ship Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more professional look
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div {
        background-color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/cargo-ship.png", width=80)
    st.title("Detection Settings")
    
    method = st.radio(
        "Region Proposal Method",
        ["fast", "quality"],
        help="Fast is quicker but less accurate. Quality is slower but more precise."
    )
    
    conf_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.8, 
        step=0.05,
        help="Higher values mean fewer but more confident detections"
    )

# Main content
st.title("Ship Detection System")
st.markdown("Upload an image to detect ships using Selective Search + Faster R-CNN")

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg", "webp"])

# Process the image
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        with st.spinner("Processing image..."):
            # Show progress bar
            progress_bar = st.progress(0)
            
            # Region proposal
            progress_bar.progress(25)
            st.info("Running selective search...")
            boxes = find_regions(image_cv, method=method)
            
            # Detection
            progress_bar.progress(50)
            st.info("Detecting ships...")
            result = detect_ships(image_cv.copy(), boxes, confidence_threshold=conf_threshold)
            
            # Finalize
            progress_bar.progress(100)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_column_width=True)
            
            # Remove progress bar and info messages
            progress_bar.empty()
            
    st.success("Detection complete!")
    
    
    # Note about accuracy
    st.info("Note: Detection accuracy depends on image complexity and selected parameters.")

else:
    # Show placeholder when no image is uploaded
    st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px;">
            <h3>Upload an image to begin detection</h3>
            <p>Supported formats: JPG, PNG, JPEG, WEBP</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Ship Detection System powered by Selective Search + Faster R-CNN")
