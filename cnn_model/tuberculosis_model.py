import os
import sys
import base64
import gc
from io import BytesIO
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the current folder to the system path so we can import 'heatmap' directly
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    import heatmap
    generate_heatmap = heatmap.generate_heatmap
    print("SUCCESS: E-Afya Heatmap module linked.")
except Exception as e:
    print(f"WARNING: Heatmap import failed ({e}). Visuals will be null.")
    def generate_heatmap(*args, **kwargs): return None

PTH_PATH = os.path.join(BASE_DIR, "tb_classifier.pth")

def load_model():
    """Initializes ResNet18 and loads trained weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(PTH_PATH):
        print(f"CRITICAL: Weights not found at {PTH_PATH}")
        return None

    checkpoint = torch.load(PTH_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    del checkpoint
    gc.collect()
    return model

model = load_model()

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
    """Processes image and returns prediction + visual heatmap."""
    try:
        gc.collect()

        if not base64_image:
            return {"status": "error", "message": "No image data."}

        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(original_image).unsqueeze(0)

        if model is None:
            return {"status": "error", "message": "Model weights missing."}

        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        prob_values, indices = torch.max(probabilities, 1)
        pred_idx = indices.item()
        confidence = prob_values.item()

        prediction = class_names[pred_idx]

        heatmap_base64 = None
        if generate_heatmap.__module__ != 'NoneType' and 'heatmap' in sys.modules:
            try:
                heatmap_base64 = generate_heatmap(model, img_tensor, original_image, pred_idx)
            except Exception as heatmap_err:
                print(f"HEATMAP LOGIC ERROR: {str(heatmap_err)}")

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%",
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        print(f"HANDLER CRASH: {str(e)}")
        return {"status": "error", "message": f"Server Error: {str(e)}"}