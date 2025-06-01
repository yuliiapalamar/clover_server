import io
import json
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms, models

# Завантаження класів
with open('class_to_idx.json', encoding='utf-8') as f:
    idx_to_class = {v: k for k, v in json.load(f).items()}

# Модель
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

# Трансформація
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_idx = outputs.argmax(1).item()
        predicted_class = idx_to_class[predicted_idx]
    return {"predicted_class": predicted_class}