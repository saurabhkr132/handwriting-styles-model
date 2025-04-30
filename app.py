import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Define Generator model
class Generator(nn.Module):
    def __init__(self, num_classes=1, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, label_embed], dim=1)
        return self.model(z)

# Load char-to-index
with open("char_to_index.json") as f:
    char_to_index = json.load(f)

device = torch.device("cpu")
num_classes = len(char_to_index)

# Load model
generator = Generator(num_classes=num_classes).to(device)
generator.load_state_dict(torch.load("saurabhkr132_generator.pth", map_location=device))
generator.eval()

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CharacterInput(BaseModel):
    character: str

@app.post("/generate")
def generate_image(data: CharacterInput):
    char = data.character
    if char not in char_to_index:
        raise HTTPException(status_code=400, detail=f"Character '{char}' not in dataset.")

    label = char_to_index[char]
    z = torch.randn(1, 64, 1, 1)
    label_tensor = torch.tensor([label], dtype=torch.long)
    with torch.no_grad():
        output = generator(z, label_tensor).squeeze().numpy()
        image = ((output + 1) / 2 * 255).astype("uint8")
        pil_image = Image.fromarray(image)

    # Convert to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"status": "success", "image_base64": img_base64}

# For local testing
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
