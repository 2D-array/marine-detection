import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from utils.selective_search import find_regions
from utils.prediction import detect_ships

st.set_page_config(page_title="Ship Detector", layout="centered")
st.title("🚢 Ship Detection using Selective Search + CNN")

uploaded_file = st.file_uploader("Upload a scene image", type=["jpg", "png", "jpeg"])

method = st.selectbox("Select Region Proposal Method", ["fast", "quality"])
conf_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, step=0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image", use_column_width=True)

    st.write("🔍 Running Selective Search and Detection...")
    boxes = find_regions(image_cv, method=method)
    result = detect_ships(image_cv.copy(), boxes, confidence_threshold=conf_threshold)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.image(result_rgb, caption="Detected Ships", use_column_width=True)
