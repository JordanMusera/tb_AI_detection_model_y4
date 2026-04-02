import os
import sys
import base64
from io import BytesIO
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms

# --- 1. SOLVE PATHING & IMPORTS ---
# Get the absolute path of 'cnn_model' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the Root ('tb_AI_detection_model_y4')
ROOT_DIR = os.path.dirname(BASE_DIR)

# Add Root to sys.path so Python can find the 'utils1' folder
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the heatmap utility from your utils1 folder
try:
    from utils1.heatmap import generate_heatmap
except ImportError:
    print("Warning: utils1.heatmap not found. Heatmaps will be disabled.")
    def generate_heatmap(*args, **kwargs): return None

# Handle truncated images (common in some medical datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 2. MODEL CONFIGURATION ---
# Points directly to the file in the same folder as this script
PTH_PATH = os.path.join(BASE_DIR, "tb_classifier.pth")

def load_model():
    """Initializes ResNet18 and loads trained weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(PTH_PATH):
        print(f"CRITICAL ERROR: Weight file missing at {PTH_PATH}")
        # List files for debugging in Render logs
        print(f"Files in {BASE_DIR}: {os.listdir(BASE_DIR)}")
        return None

    # Load to CPU for Render's stability
    model.load_state_dict(torch.load(PTH_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Global model instance for efficiency (loads once on startup)
model = load_model()

# Image Preprocessing (Must match your training script exactly)
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
    """Main function called by api.py to process images."""
    try:
        if not base64_image:
            return {"status": "error", "message": "Empty image string."}

        # Handle Base64 Data URL prefix (e.g., 'data:image/png;base64,')
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        # Decode and Process Image
        image_bytes = base64.b64decode(base64_image)
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(original_image).unsqueeze(0)

        if model is None:
            return {"status": "error", "message": "Model not loaded on server."}

        # Run Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(1).item()

            prediction = class_names[pred_idx]
            confidence = probabilities[0][pred_idx].item()

        # Generate Visual Heatmap (Grad-CAM)
        heatmap_base64 = generate_heatmap(model, img_tensor, original_image, pred_idx)

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        print(f"HANDLER ERROR: {str(e)}")
        return {"status": "error", "message": f"Server processing failed: {str(e)}"}