"""Phase 8: Production Inference Module for EcoVision.

Provides CLI and programmatic prediction on waste images using the final MobileNetV2 model.
Maintains exact preprocessing alignment with training and validation pipelines:
RGB -> direct resize to 224x224 (Bilinear) -> ToTensor() -> ImageNet Normalize.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import create_mobilenetv2_transfer

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_classes_json(models_dir: Path) -> Path:
    """Save class names in ImageFolder alphabetical order to models/classes.json."""
    classes_path = models_dir / "classes.json"
    if not classes_path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(classes_path, "w") as f:
            json.dump(CLASS_NAMES, f, indent=2)
    return classes_path


def load_threshold(models_dir: Path) -> float:
    """Load optimized confidence threshold from models/threshold.json."""
    thresh_path = models_dir / "threshold.json"
    if thresh_path.exists():
        with open(thresh_path, "r") as f:
            data = json.load(f)
            return float(data.get("chosen_threshold", 0.80))
    return 0.80


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image matching training cache and evaluation exactly.

    Pipeline:
      1. Convert to RGB
      2. Direct resize to 224x224 (Bilinear interpolation, NO center crop)
      3. Convert to FloatTensor in [0, 1]
      4. Standard ImageNet channel normalization
      5. Add batch dimension -> (1, 3, 224, 224)
    """
    img_rgb = pil_image.convert("RGB")
    img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = transforms.ToTensor()(img_resized)
    tensor = norm(tensor)
    return tensor.unsqueeze(0)


# Cached global model and device for fast repeated programmatic calls
_CACHED_MODEL: Optional[nn.Module] = None
_CACHED_DEVICE: Optional[torch.device] = None


def get_model(device: Optional[torch.device] = None) -> Tuple[nn.Module, torch.device]:
    """Retrieve or load cached MobileNetV2 final model."""
    global _CACHED_MODEL, _CACHED_DEVICE
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CACHED_MODEL is None or _CACHED_DEVICE != device:
        models_dir = project_root / "models"
        ckpt_path = models_dir / "mobilenetv2_final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

        model = create_mobilenetv2_transfer(num_classes=len(CLASS_NAMES), dropout_p=0.3, pretrained=False)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _CACHED_MODEL = model
        _CACHED_DEVICE = device
        ensure_classes_json(models_dir)

    return _CACHED_MODEL, device


def predict_image(
    image_input: Union[Image.Image, str, Path],
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    threshold: Optional[float] = None,
) -> Dict:
    """Predict waste class for a PIL Image or image file path.

    Returns dict containing:
      - 'predicted_class': Top predicted class name
      - 'confidence': Probability score of top class [0.0, 1.0]
      - 'uncertain': True if confidence is below confidence threshold
      - 'threshold': Confidence threshold applied
      - 'probabilities': Dict of class names to probabilities
      - 'top3': List of (class_name, probability) tuples
    """
    models_dir = project_root / "models"
    if threshold is None:
        threshold = load_threshold(models_dir)

    if model is None:
        model, device = get_model(device)
    elif device is None:
        device = next(model.parameters()).device

    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
    else:
        pil_img = image_input

    tensor = preprocess_image(pil_img).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [(CLASS_NAMES[i], round(float(probs[i]), 4)) for i in top3_indices]

    prob_dict = {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)}

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "uncertain": bool(confidence < threshold),
        "threshold": round(threshold, 4),
        "probabilities": prob_dict,
        "top3": top3,
    }


def main():
    parser = argparse.ArgumentParser(description="EcoVision Waste Classification CLI")
    parser.add_argument("image_path", type=str, help="Path to image file (.jpg, .png, etc.)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom confidence threshold")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: File not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    models_dir = project_root / "models"
    ensure_classes_json(models_dir)

    res = predict_image(img_path, threshold=args.threshold)

    print("\n" + "=" * 50)
    print("EcoVision Classification Result")
    print("=" * 50)
    print(f"Image     : {img_path.name}")
    print(f"Predicted : {res['predicted_class']}")
    print(f"Confidence: {res['confidence']*100:.2f}%")
    if res["uncertain"]:
        print(f"Status    : UNCERTAIN (below threshold {res['threshold']*100:.1f}%)")
    else:
        print(f"Status    : ACCEPTED (confidence >= {res['threshold']*100:.1f}%)")

    print("\nTop-3 Predictions:")
    for rank, (cls, p) in enumerate(res["top3"], 1):
        print(f"  {rank}. {cls:<10}: {p*100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
