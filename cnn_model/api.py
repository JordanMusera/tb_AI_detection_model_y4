import os
import sys
import importlib.util  # Essential for direct file loading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 1. INITIALIZE APP ---
app = FastAPI(title="E-Afya Disease Detection System API")

# --- 2. CORS CONFIGURATION ---
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

# --- 4. CORE ENGINE (DIRECT FILE LOADER) ---
def run_model(model_name: str, base64_image: str):
    try:
        # Get the absolute path to the directory containing THIS file (api.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Prepare file details
        clean_model_name = model_name.lower().strip()
        module_name = f"{clean_model_name}_model"
        file_path = os.path.join(current_dir, f"{module_name}.py")

        # --- THE DIRECT LOAD LOGIC ---
        # This ignores Python's internal search and grabs the file directly from the disk
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)

        # Register the module so internal imports inside the model work
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Check for the required handler function
        if hasattr(module, "handler"):
            return module.handler(base64_image)
        else:
            return {
                "status": "error",
                "message": f"Critical: 'handler()' function missing in {module_name}.py"
            }

    except Exception as e:
        # Detailed error reporting for Render logs
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

# --- 5. ROUTES ---

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

# --- 6. RENDER EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)