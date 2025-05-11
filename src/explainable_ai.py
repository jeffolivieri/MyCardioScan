# src/explainable_ai.py
from pytorch_grad_cam import GradCAM
import torch
import numpy as np

class BinaryClassifierOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output  # No indexing

def generate_gradcam(model, input_tensor, target_layer):
    """
    Generate GradCAM for binary classification model.
    """

    cam = GradCAM(model=model, target_layers=[target_layer])

    # Explicitly set custom target for scalar output
    targets = [BinaryClassifierOutputTarget()]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    mid_slice_idx = input_tensor.shape[2] // 2
    grayscale_slice = grayscale_cam[0, mid_slice_idx, :, :]

    return grayscale_slice
