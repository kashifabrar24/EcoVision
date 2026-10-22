---
title: EcoVision
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

EcoVision is an automated recyclable waste classification web application powered by a fine-tuned MobileNetV2.
The model categorizes waste images into six distinct classes: cardboard, glass, metal, paper, plastic, and trash.
It incorporates a selective classification decision threshold to safely route low-confidence predictions to human review.
Pretrained on ImageNet and fine-tuned on the TrashNet dataset using data augmentation and stratified group splits.
Note: Trained on clean single-item photos on plain backgrounds; real-world mixed campus waste may be more challenging.
