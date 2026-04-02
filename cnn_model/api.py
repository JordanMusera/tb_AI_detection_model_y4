import os
import sys
import importlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. INITIALIZE APP ---
app = FastAPI(title="E-Afya Disease Detection System API")

# --- 2. CORS CONFIGURATION ---
# Added common Render and Localhost origins for your frontend
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

# --- 3. DATA MODELS ---
class ImageInput(BaseModel):
    model: str = "tuberculosis"
    image: str # Base64 string

# --- 4. CORE ENGINE (DYNAMIC MODULE LOADER) ---
def run_model(model_name: str, base64_image: str):
    try:
        # Get the absolute path to the directory containing THIS file (api.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # FORCE Python to look inside cnn_model for your .py model files
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Linux (Render) is case-sensitive. We force lowercase to prevent errors.
        clean_model_name = model_name.lower().strip()
        module_name = f"{clean_model_name}_model"

        # Dynamically import the module (e.g., tuberculosis_model.py)
        if module_name in sys.modules:
            # Reload to ensure any changes to the .pth loading are fresh
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)

        # Check if the model file has the required 'handler' function
        if hasattr(module, "handler"):
            return module.handler(base64_image)
        else:
            return {
                "status": "error",
                "message": f"Critical: 'handler()' function missing in {module_name}.py"
            }

    except ModuleNotFoundError:
        # Debugging info for Render Logs
        available_files = os.listdir(current_dir) if os.path.exists(current_dir) else "Dir not found"
        return {
            "status": "error",
            "message": f"Model file '{module_name}.py' not found on server.",
            "debug_info": {
                "searched_path": current_dir,
                "files_present": available_files
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Processing Error: {str(e)}"
        }

# --- 5. ROUTES ---

@app.get("/")
async def health_check():
    """Verify the server is awake on Render."""
    return {
        "status": "online",
        "project": "E-Afya AI Detection",
        "region": "Render Cloud"
    }

@app.post("/post")
async def general_predict(data: ImageInput):
    """General endpoint used by the frontend."""
    if not data.image:
        raise HTTPException(status_code=400, detail="No base64 image data provided")
    return run_model(data.model, data.image)

@app.post("/predict/tuberculosis")
async def predict_tb_direct(data: ImageInput):
    """Direct shortcut for TB detection."""
    return run_model("tuberculosis", data.image)

# --- 6. LOCAL EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    # Note: Render overrides this port with the $PORT env variable
    uvicorn.run(app, host="0.0.0.0", port=10000)