"""Phase 9: EcoVision Gradio Web Application.

Provides an interactive web interface for real-time waste classification into
6 categories (cardboard, glass, metal, paper, plastic, trash) using the fine-tuned
MobileNetV2 model with selective classification confidence thresholding.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
from PIL import Image
import gradio as gr

from src.predict import predict_image

REPO_ROOT = Path(__file__).resolve().parent
DEMO_DIR = REPO_ROOT / "demo_examples"

# Predefined test examples for demonstration (one per class, all < 200 KB)
EXAMPLE_PATHS = [
    [str(DEMO_DIR / "cardboard101.jpg")],
    [str(DEMO_DIR / "glass113.jpg")],
    [str(DEMO_DIR / "metal100.jpg")],
    [str(DEMO_DIR / "paper10.jpg")],
    [str(DEMO_DIR / "plastic114.jpg")],
    [str(DEMO_DIR / "trash1.jpg")],
]


def classify_waste(image: Union[Image.Image, None]) -> Tuple[Dict[str, float], str]:
    """Run EcoVision classification on an input PIL image.

    Returns:
        tuple: (dict of class probabilities for gr.Label, status text string)
    """
    if image is None:
        return {}, "Please upload an image to classify."

    result = predict_image(image)
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    is_uncertain = result["uncertain"]

    if is_uncertain:
        status_line = f"Uncertain: confidence below threshold, closest class: {pred_class}"
    else:
        status_line = f"Predicted: {pred_class} (confidence {confidence * 100:.1f}%)"

    return result["probabilities"], status_line


def build_app() -> gr.Interface:
    """Build and configure the Gradio Interface."""
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


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
