"""EcoVision Streamlit Application.

Self-contained Streamlit web demo for automated recyclable waste classification.
Runs entirely on CPU with zero external project dependencies outside standard libraries.
"""

import json
from pathlib import Path
from typing import Dict, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as tv_models
import streamlit as st

# Setup paths relative to this file so execution works from any CWD
APP_DIR = Path(__file__).resolve().parent

# Load class names from local classes.json
CLASSES_PATH = APP_DIR / "classes.json"
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load decision threshold from local threshold.json
THRESHOLD_PATH = APP_DIR / "threshold.json"
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        THRESH_DATA = json.load(f)
        DECISION_THRESHOLD = float(THRESH_DATA.get("chosen_threshold", 0.80))
else:
    DECISION_THRESHOLD = 0.80

DEMO_DIR = APP_DIR / "demo_examples"

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORM_TRANSFORM = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess PIL Image matching space/app.py exactly."""
    img_rgb = pil_image.convert("RGB")
    img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
    tensor = transforms.ToTensor()(img_resized)
    tensor = NORM_TRANSFORM(tensor)
    return tensor.unsqueeze(0)


def build_model(num_classes: int = 6, dropout_p: float = 0.3) -> nn.Module:
    """Instantiate MobileNetV2 architecture with matching classifier head."""
    model = tv_models.mobilenet_v2(weights=None)
    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    return model


@st.cache_resource
def load_model() -> nn.Module:
    """Load cached model on CPU."""
    model_path = APP_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing weights file: {model_path}")
    model = build_model(num_classes=len(CLASS_NAMES), dropout_p=0.3)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()
    return model


def predict_image(
    image_input: Union[Image.Image, str, Path],
    model: nn.Module,
) -> Dict:
    """Run inference on PIL Image or image file path.

    Returns dict containing:
      - 'predicted_class': Top predicted class name
      - 'confidence': Top probability score [0.0, 1.0]
      - 'uncertain': Boolean flag indicating if confidence < threshold
      - 'threshold': Confidence threshold applied
      - 'probabilities': Dict mapping class names to probability scores
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
    else:
        pil_img = image_input

    tensor = preprocess_image(pil_img)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    is_uncertain = bool(confidence < DECISION_THRESHOLD)

    prob_dict = {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)}

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "uncertain": is_uncertain,
        "threshold": round(DECISION_THRESHOLD, 4),
        "probabilities": prob_dict,
    }


def main():
    st.set_page_config(page_title="EcoVision", page_icon="♻️", layout="centered")

    st.title("EcoVision")
    st.write(
        "Automated recyclable waste classification into 6 categories: "
        "**cardboard, glass, metal, paper, plastic, and trash**."
    )

    st.info(
        "**Note:** The model is trained on clean single-item photos on a plain background, "
        "so real campus waste may be harder."
    )

    # Initialize session state for selected image and source name
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "source_name" not in st.session_state:
        st.session_state.source_name = None

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Section 1: Example Images
    st.write("### Try an Example Image")
    example_items = [
        ("Cardboard", "cardboard101.jpg"),
        ("Glass", "glass113.jpg"),
        ("Metal", "metal100.jpg"),
        ("Paper", "paper10.jpg"),
        ("Plastic", "plastic114.jpg"),
        ("Trash", "trash1.jpg"),
    ]

    cols = st.columns(6)
    for col, (label, filename) in zip(cols, example_items):
        if col.button(label, use_container_width=True):
            ex_path = DEMO_DIR / filename
            if ex_path.exists():
                st.session_state.selected_image = Image.open(ex_path)
                st.session_state.source_name = f"Example: {filename}"

    # Section 2: Upload Option
    st.write("### Or Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        st.session_state.selected_image = Image.open(uploaded_file)
        st.session_state.source_name = f"Uploaded: {uploaded_file.name}"

    # Section 3: Prediction and Visualizations
    if st.session_state.selected_image is not None:
        st.write("---")
        img = st.session_state.selected_image

        col_img, col_pred = st.columns([1, 1])

        with col_img:
            st.image(img, caption=st.session_state.source_name, use_container_width=True)

        with col_pred:
            res = predict_image(img, model)
            pred_class = res["predicted_class"]
            conf = res["confidence"]
            is_uncertain = res["uncertain"]

            if is_uncertain:
                st.warning(
                    f"Uncertain: confidence below threshold, closest class: {pred_class}"
                )
            else:
                st.success(
                    f"Predicted: {pred_class} (confidence {conf * 100:.1f}%)"
                )

            st.write(f"**Confidence Score:** {conf * 100:.2f}%")
            st.write(f"**Operating Threshold:** {DECISION_THRESHOLD * 100:.1f}%")

        # Bar chart of all 6 class probabilities
        st.write("### Class Probabilities")
        chart_data = pd.DataFrame(
            {
                "Class": list(res["probabilities"].keys()),
                "Probability": list(res["probabilities"].values()),
            }
        ).set_index("Class")

        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
