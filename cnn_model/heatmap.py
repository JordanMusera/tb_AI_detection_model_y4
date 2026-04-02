import numpy as np
from PIL import Image
from io import BytesIO
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_heatmap(model, img_tensor, original_image, pred_idx):
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]

    img_np = np.array(original_image.resize((224, 224))) / 255.0
    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    buffer = BytesIO()
    Image.fromarray(heatmap).save(buffer, format="JPEG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

    return heatmap_base64
