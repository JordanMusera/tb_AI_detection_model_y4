import os, base64
from io import BytesIO
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms

from utils1.heatmap import generate_heatmap

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PTH_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "cnn_model", "tb_classifier.pth"))

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(PTH_PATH):
        raise FileNotFoundError(f"Model weight file not found at: {PTH_PATH}")

    model.load_state_dict(torch.load(PTH_PATH, map_location=torch.device('cpu')))
    model.to("cpu")
    model.eval()
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
    try:
        if not base64_image:
            return {"status": "error", "message": "No image data provided"}

        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        img_tensor = transform(original_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            pred_idx = outputs.argmax(1).item()
            prediction = class_names[pred_idx]

        heatmap_base64 = generate_heatmap(model, img_tensor, original_image, pred_idx)

        return {
            "status": "success",
            "prediction": prediction,
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        print("MODEL ERROR:", repr(e))
        return {"status": "error", "message": str(e)}