import os, base64
import sys
from io import BytesIO
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms

# Ensure the parent directory is in path so it can find 'utils1'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from utils1.heatmap import generate_heatmap
except ImportError:
    # Fallback for different folder structures on Render
    from heatmap import generate_heatmap

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- PATH LOGIC ---
# We look for the .pth file in the SAME folder as this script
PTH_PATH = os.path.join(BASE_DIR, "tb_classifier.pth")

def load_model():
    # resnet18 architecture to match your training
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(PTH_PATH):
        # Debugging print for Render logs
        print(f"CRITICAL: .pth not found at {PTH_PATH}")
        print(f"Available files in {BASE_DIR}: {os.listdir(BASE_DIR)}")
        raise FileNotFoundError(f"Model weight file not found at: {PTH_PATH}")

    # Load to CPU (Required for Render Free Tier)
    model.load_state_dict(torch.load(PTH_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Global model instance so it only loads ONCE when the server starts
model = load_model()

# Standard ResNet transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["Normal", "Tuberculosis"]

def handler(base64_image: str):
    try:
        if not base64_image:
            return {"status": "error", "message": "No image data provided"}

        # Strip base64 header if present
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Prepare image for ResNet
        img_tensor = transform(original_image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            # Softmax to get confidence (optional, but good for E-Afya UI)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(1).item()
            confidence = probabilities[0][pred_idx].item()

            prediction = class_names[pred_idx]

        # Generate the Grad-CAM heatmap
        heatmap_base64 = generate_heatmap(model, img_tensor, original_image, pred_idx)

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        print("MODEL ERROR:", repr(e))
        return {"status": "error", "message": f"AI Processing Error: {str(e)}"}