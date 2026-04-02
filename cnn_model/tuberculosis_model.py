import os
import sys
import base64
import gc
from io import BytesIO
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms

# Handle truncated images (common in some medical datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 1. PATHING & IMPORTS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Since heatmap.py is now in the same 'cnn_model' folder
    from .heatmap import generate_heatmap
    print("SUCCESS: Heatmap module loaded.")
except ImportError as e:
    print(f"WARNING: Heatmap module not found ({e}). Visuals disabled.")
    def generate_heatmap(*args, **kwargs): return None

# --- 2. MODEL CONFIGURATION ---
PTH_PATH = os.path.join(BASE_DIR, "tb_classifier.pth")

def load_model():
    """Initializes ResNet18 and loads trained weights efficiently."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(PTH_PATH):
        print(f"CRITICAL ERROR: Weight file missing at {PTH_PATH}")
        return None

    # Load to CPU and immediately set to eval mode
    checkpoint = torch.load(PTH_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    # Clean up memory immediately after loading
    del checkpoint
    gc.collect()
    return model

# Global model instance
model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["Normal", "Tuberculosis"]

# --- 3. THE HANDLER FUNCTION ---
def handler(base64_image: str):
    """Processes image and returns prediction + visual heatmap."""
    try:
        # Clear memory before starting heavy AI work
        gc.collect()

        if not base64_image:
            return {"status": "error", "message": "Empty image string received."}

        # Handle Base64 Data URL prefix
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        # Decode Image
        image_bytes = base64.b64decode(base64_image)
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(original_image).unsqueeze(0)

        if model is None:
            return {"status": "error", "message": "Model weights failed to load on server."}

        # --- Prediction Logic ---
        # Note: We do NOT use 'with torch.no_grad()' here because
        # generate_heatmap needs gradients to trace the decision.
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get Confidence and Index
        prob_values, indices = torch.max(probabilities, 1)
        pred_idx = indices.item()
        confidence = prob_values.item()

        prediction = class_names[pred_idx]

        # --- Heatmap Generation ---
        heatmap_base64 = None
        try:
            # Only run if heatmap module exists
            if generate_heatmap.__name__ != 'NoneType':
                heatmap_base64 = generate_heatmap(model, img_tensor, original_image, pred_idx)
        except Exception as heatmap_err:
            print(f"HEATMAP FAILED: {str(heatmap_err)}")
            # We don't crash the whole request if only the heatmap fails

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        print(f"HANDLER ERROR: {str(e)}")
        return {"status": "error", "message": f"Processing failed: {str(e)}"}