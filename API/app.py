from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MODEL.model import get_model
from DATA_PREPROCESSING.preprocessing import get_val_transforms

app = FastAPI(title="PrintGuard AI - Quality Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'TRAINING', 'best_model.pth')
CLASSES = ['Excellent', 'Good', 'Fair', 'Poor']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

def load_model():
    global model
    if model is None:
        model = get_model(num_classes=len(CLASSES)).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file {MODEL_PATH} not found. Running with untrained weights.")
    return model

@app.get("/")
async def root():
    return {"message": "PrintGuard AI Quality Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        transform = get_val_transforms()
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        loaded_model = load_model()
        with torch.no_grad():
            outputs = loaded_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)

        result = {
            "prediction": CLASSES[preds[0].item()],
            "confidence": float(confidence[0].item()),
            "all_scores": {CLASSES[i]: float(probabilities[0][i].item()) for i in range(len(CLASSES))}
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
