import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import uvicorn
from contextlib import asynccontextmanager

from model import CVSClassifier

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "convenext_tiny/cvs_endoscapes_convnext_best.pth"
THRESHOLD = 0.5

# Model instance
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        model = CVSClassifier(dropout=0.0, freeze_backbone=False, num_outputs=1)
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        print(f"Loaded model successfully to {DEVICE} from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    # Cleanup on shutdown
    print("Shutting down the server")

app = FastAPI(lifespan=lifespan)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inference transform tailored to ConvNeXt-Tiny Endoscapes configuration
infer_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file, preprocesses it, and runs it through the detection model."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
         return JSONResponse(status_code=400, content={"error": "File must be an image (.jpg, .jpeg, .png)."})
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = infer_transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            # Apply sigmoid since single output
            prob = torch.sigmoid(output).item()
            
        achieved = bool(prob >= THRESHOLD)
        
        return {
            "probability": prob,
            "achieved": achieved,
            "threshold": THRESHOLD
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount frontend docs directory (HTML, CSS, JS)
try:
    app.mount("/", StaticFiles(directory="docs", html=True), name="docs")
except Exception as e:
    print("Warning: docs directory not found, skipping frontend mount.")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
