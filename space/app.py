"""EcoVision Hugging Face Space Application.

Self-contained Gradio web demo for automated recyclable waste classification.
Runs entirely on CPU with zero external project dependencies outside standard libraries.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as tv_models
import gradio as gr

APP_DIR = Path(__file__).resolve().parent

# Load class names from local classes.json
CLASSES_PATH = APP_DIR / "classes.json"
if CLASSES_PATH.exists():
    with open(CLASSES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load decision threshold from local threshold.json
THRESHOLD_PATH = APP_DIR / "threshold.json"
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, "r") as f:
        THRESH_DATA = json.load(f)
        DECISION_THRESHOLD = float(THRESH_DATA.get("chosen_threshold", 0.80))
else:
    DECISION_THRESHOLD = 0.80

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing transform pipeline: RGB -> 224x224 Bilinear -> ToTensor -> Normalize
NORM_TRANSFORM = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess PIL Image matching training pipeline exactly."""
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


# Lazy-loaded model instance
_MODEL = None


def get_model() -> nn.Module:
    """Load cached model on CPU."""
    global _MODEL
    if _MODEL is None:
        model_path = APP_DIR / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing weights file: {model_path}")
        model = build_model(num_classes=len(CLASS_NAMES), dropout_p=0.3)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to("cpu")
        model.eval()
        _MODEL = model
    return _MODEL


def predict_image(image_input: Union[Image.Image, str, Path]) -> Dict:
    """Run inference on PIL Image or image file path.

    Returns dict containing:
      - 'predicted_class': Top predicted class name
      - 'confidence': Top probability score [0.0, 1.0]
      - 'uncertain': Boolean flag indicating if confidence < threshold
      - 'threshold': Confidence threshold applied
      - 'probabilities': Dict mapping class names to probability scores
    """
    model = get_model()

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


def classify_waste(image: Union[Image.Image, None]) -> Tuple[Dict[str, float], str]:
    """Gradio handler for waste image classification."""
    if image is None:
        return {}, "Please upload an image to classify."

    res = predict_image(image)
    pred_class = res["predicted_class"]
    conf = res["confidence"]

    if res["uncertain"]:
        status_line = f"Uncertain: confidence below threshold, closest class: {pred_class}"
    else:
        status_line = f"Predicted: {pred_class} (confidence {conf * 100:.1f}%)"

    return res["probabilities"], status_line


# Collect example paths from local demo_examples/
DEMO_DIR = APP_DIR / "demo_examples"
EXAMPLE_PATHS = [
    [str(DEMO_DIR / "cardboard101.jpg")],
    [str(DEMO_DIR / "glass113.jpg")],
    [str(DEMO_DIR / "metal100.jpg")],
    [str(DEMO_DIR / "paper10.jpg")],
    [str(DEMO_DIR / "plastic114.jpg")],
    [str(DEMO_DIR / "trash1.jpg")],
]


def build_app() -> gr.Interface:
    """Build Gradio UI interface."""
    title = "EcoVision"
    description = (
        "Automated waste sorting into 6 recyclable categories (cardboard, glass, metal, "
        "paper, plastic, trash) using a fine-tuned MobileNetV2 architecture with selective "
        "classification rejection.\n\n"
        "**Real-World Generalization Note:** The model was trained on clean single-item photos "
        "on a plain background, so real campus waste may be harder."
    )

    demo = gr.Interface(
        fn=classify_waste,
        inputs=gr.Image(type="pil", label="Upload Waste Image"),
        outputs=[
            gr.Label(num_top_classes=6, label="Class Probabilities"),
            gr.Textbox(label="Decision Status", interactive=False),
        ],
        title=title,
        description=description,
        examples=[ex for ex in EXAMPLE_PATHS if Path(ex[0]).exists()],
        flagging_mode="never",
    )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
