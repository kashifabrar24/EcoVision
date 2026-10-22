"""EcoVision Streamlit Application.

Self-contained Streamlit web demo for automated recyclable waste classification.
Runs entirely on CPU with zero external project dependencies outside standard libraries.
"""

import base64
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


def get_image_base64(path: Path) -> str:
    """Return base64 string for embedding local thumbnail images."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    st.set_page_config(layout="wide", page_title="EcoVision", page_icon="♻️")

    # Light CSS for clean tool aesthetics, consistent spacing, and subtle borders
    custom_css = """
    <style>
    .main .block-container {
        max-width: 1080px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    img {
        border-radius: 6px;
    }
    div[data-testid="stButton"] button {
        border-radius: 6px;
        padding: 0.25rem 0.5rem;
        font-size: 0.85rem;
    }
    .thumb-card {
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 6px;
        transition: border-color 0.15s ease;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Header
    st.title("EcoVision")
    st.write(
        "Pick a sample or upload a photo of a single item. If the model is less than 80% sure, "
        "it says so instead of guessing."
    )

    with st.expander("About this model"):
        st.markdown(
            "- **Architecture:** MobileNetV2 with transfer learning\n"
            "- **Dataset:** TrashNet (2,521 images after cleaning, 6 classes, 379 test images)\n"
            "- **Test accuracy:** 89.71%\n"
            "- **Test macro-F1:** 0.8647\n"
            "- **Confidence threshold:** 0.80 (chosen on validation)\n"
            "- **Repository:** [https://github.com/kashifabrar24/EcoVision](https://github.com/kashifabrar24/EcoVision)\n\n"
            "**Note on campus waste:** The model is trained on clean single-item photos on a plain background, "
            "so real campus waste may be harder."
        )

    st.write("")

    # Session state initialization
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "source_name" not in st.session_state:
        st.session_state.source_name = None
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Two panels: Left (Choose an image) & Right (Result)
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("1. Choose an image")

        # 3x2 Grid for sample images
        example_items = [
            ("Cardboard", "cardboard101.jpg"),
            ("Glass", "glass113.jpg"),
            ("Metal", "metal100.jpg"),
            ("Paper", "paper10.jpg"),
            ("Plastic", "plastic114.jpg"),
            ("Trash", "trash1.jpg"),
        ]

        # Row 1 (first 3)
        row1_cols = st.columns(3)
        for col, (label, filename) in zip(row1_cols, example_items[:3]):
            with col:
                ex_path = DEMO_DIR / filename
                if ex_path.exists():
                    is_selected = (st.session_state.selected_sample == filename)
                    border = "2px solid #2F7D4F" if is_selected else "1px solid rgba(128, 128, 128, 0.25)"
                    b64_img = get_image_base64(ex_path)
                    st.markdown(
                        f'<div class="thumb-card" style="border: {border};">'
                        f'<img src="data:image/jpeg;base64,{b64_img}" style="width: 100%; aspect-ratio: 1; object-fit: cover; display: block;" />'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(label, key=f"btn_{filename}", use_container_width=True):
                        st.session_state.selected_image = Image.open(ex_path)
                        st.session_state.source_name = f"Example: {filename}"
                        st.session_state.selected_sample = filename
                        st.rerun()

        st.write("")

        # Row 2 (next 3)
        row2_cols = st.columns(3)
        for col, (label, filename) in zip(row2_cols, example_items[3:]):
            with col:
                ex_path = DEMO_DIR / filename
                if ex_path.exists():
                    is_selected = (st.session_state.selected_sample == filename)
                    border = "2px solid #2F7D4F" if is_selected else "1px solid rgba(128, 128, 128, 0.25)"
                    b64_img = get_image_base64(ex_path)
                    st.markdown(
                        f'<div class="thumb-card" style="border: {border};">'
                        f'<img src="data:image/jpeg;base64,{b64_img}" style="width: 100%; aspect-ratio: 1; object-fit: cover; display: block;" />'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(label, key=f"btn_{filename}", use_container_width=True):
                        st.session_state.selected_image = Image.open(ex_path)
                        st.session_state.source_name = f"Example: {filename}"
                        st.session_state.selected_sample = filename
                        st.rerun()

        st.write("")
        st.write("**Or upload your own image:**")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.session_state.selected_image = Image.open(uploaded_file)
            st.session_state.source_name = f"Uploaded: {uploaded_file.name}"
            st.session_state.selected_sample = None

    with right_col:
        st.subheader("2. Result")

        if st.session_state.selected_image is None:
            st.info("Pick a sample from the left or upload an image to classify.")
        else:
            img = st.session_state.selected_image
            st.image(img, caption=st.session_state.source_name, use_container_width=True)

            res = predict_image(img, model)
            pred_class = res["predicted_class"]
            conf = res["confidence"]
            is_uncertain = res["uncertain"]

            if is_uncertain:
                st.warning(
                    f"Uncertain: confidence below the 80% threshold, closest class: {pred_class}"
                )
            else:
                st.success(
                    f"Predicted: {pred_class} (confidence {conf * 100:.1f}%)"
                )

            # Top-3 classes as compact rows with progress bars
            sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)
            st.write("**Top-3 classes**")
            for cls_name, prob_val in sorted_probs[:3]:
                st.write(f"{cls_name} — {prob_val * 100:.1f}%")
                st.progress(min(max(float(prob_val), 0.0), 1.0))

            st.write("")
            # Small bar chart of all 6 probabilities
            st.write("**All class probabilities**")
            chart_data = pd.DataFrame(
                {
                    "Class": list(res["probabilities"].keys()),
                    "Probability": list(res["probabilities"].values()),
                }
            ).set_index("Class")
            st.bar_chart(chart_data, height=180)

            # Short caption under the result
            st.caption(
                "Note: The model is trained on clean single-item photos on a plain background, "
                "so real campus waste may be harder."
            )

    # Footer
    st.markdown("---")
    st.caption("Dataset: TrashNet (Gary Thung and Mindy Yang, MIT license)")


if __name__ == "__main__":
    main()
