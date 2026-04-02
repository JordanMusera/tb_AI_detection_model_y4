import os
import sys
import importlib.util
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="E-Afya Disease Detection System API")

origins = [
    "https://cnn-disease-detection-system-client.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    model: str = "tuberculosis"
    image: str # Base64 string

def run_model(model_name: str, base64_image: str):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        clean_model_name = model_name.lower().strip()
        module_name = f"{clean_model_name}_model"
        file_path = os.path.join(current_dir, f"{module_name}.py")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)

        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "handler"):
            return module.handler(base64_image)
        else:
            return {
                "status": "error",
                "message": f"Critical: 'handler()' function missing in {module_name}.py"
            }

    except Exception as e:
        available_files = os.listdir(current_dir) if os.path.exists(current_dir) else "Dir not found"
        return {
            "status": "error",
            "message": f"System Load Error: {str(e)}",
            "debug_info": {
                "attempted_file": f"{module_name}.py",
                "searched_path": current_dir,
                "files_present": available_files
            }
        }

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "project": "E-Afya AI Detection",
        "env": "Render Production"
    }

@app.post("/post")
async def general_predict(data: ImageInput):
    if not data.image:
        raise HTTPException(status_code=400, detail="No base64 image data provided")
    return run_model(data.model, data.image)

@app.post("/predict/tuberculosis")
async def predict_tb_direct(data: ImageInput):
    return run_model("tuberculosis", data.image)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)