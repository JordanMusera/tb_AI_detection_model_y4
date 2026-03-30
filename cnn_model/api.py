import os
import sys
import importlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CNN Disease Detection System API")

origins = [
    "https://cnn-disease-detection-system-client.onrender.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

class ImageInput(BaseModel):
    model: str = "tuberculosis"
    image: str

def run_model(model_name: str, base64_image: str):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        module_name = f"{model_name}_model"

        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)

        if hasattr(module, "handler"):
            return module.handler(base64_image)
        else:
            return {
                "status": "error",
                "message": f"Critical Error: 'handler()' function not found in {module_name}.py"
            }

    except ModuleNotFoundError as e:
        files_in_dir = os.listdir(current_dir) if os.path.exists(current_dir) else "Directory not found"
        return {
            "status": "error",
            "message": f"Model file '{model_name}_model.py' not found. Files in directory: {files_in_dir}",
            "debug_path": current_dir
        }
    except Exception as e:
        return {"status": "error", "message": f"Processing Error: {str(e)}"}

@app.get("/")
async def root():
    return {"status": "online", "project": "tb_AI_detection_model_y4"}

@app.post("/post")
async def general_post_route(data: ImageInput):
    if not data.image:
        raise HTTPException(status_code=400, detail="No image provided")
    return run_model(data.model, data.image)

@app.post("/predict/tuberculosis")
async def predict_tb(data: ImageInput):
    return run_model("tuberculosis", data.image)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)