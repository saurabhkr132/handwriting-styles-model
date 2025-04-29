import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
import gradio as gr
import os

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

# Load model once
generator = Generator(num_classes=num_classes).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Generation function
def generate_image(character):
    if character not in char_to_index:
        return f"Character '{character}' not in dataset.", None
    label = char_to_index[character]
    z = torch.randn(1, 64, 1, 1)
    label_tensor = torch.tensor([label], dtype=torch.long)
    with torch.no_grad():
        output = generator(z, label_tensor).squeeze().numpy()
        image = ((output + 1) / 2 * 255).astype("uint8")
        pil_image = Image.fromarray(image)
    return "Generated successfully", pil_image

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Handwriting GAN — Generate a handwritten character")
    char_input = gr.Textbox(label="Character")
    generate_btn = gr.Button("Generate")
    status = gr.Textbox(label="Status")
    image_output = gr.Image(label="Generated Image")

    generate_btn.click(fn=generate_image, inputs=char_input, outputs=[status, image_output])

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)