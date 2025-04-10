import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import DigitRecognizerCNN  

# ---- Load model ---- #
model = DigitRecognizerCNN()
model_path = "checkpoints/best_model_20250410_194435.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# ---- Streamlit UI ---- #
st.title(" MNIST Digit Classifier")
st.write("Draw a digit below and click **Predict** to see what the model thinks it is.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_color="#FFFFFF",
    stroke_width=15,
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to grayscale image
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("L")

        # Invert colors: white digit on black (like MNIST)
        img = ImageOps.invert(img)

        # Binarize: make black/white sharp (optional but helps)
        img = img.point(lambda x: 0 if x < 128 else 255, "1")

        # Crop to content
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Resize to 28x28 like MNIST
        img = img.resize((28, 28))

        # Show the processed image
        st.image(img, caption="Processed Digit", width=150)

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        st.markdown(f"###  Prediction: **{prediction.item()}**")
        st.markdown(f"###  Confidence: **{confidence.item() * 100:.2f}%**")
    else:
        st.warning("Please draw a digit before clicking Predict.")
