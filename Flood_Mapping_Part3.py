import os
import torch
import numpy as np
import tifffile as tiff
from flask import Flask, request, jsonify, render_template, url_for
from PIL import Image
from model import CustomDeepLabV3
import torchvision.transforms as transforms

app = Flask(_name_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "deeplabv3_d.pth"
NUM_INPUT_CHANNELS = 12
IMG_HEIGHT, IMG_WIDTH = 128, 128

model = CustomDeepLabV3(num_input_channels=NUM_INPUT_CHANNELS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Normalization 
USE_NORMALIZATION = False  
mean = [0.485] * NUM_INPUT_CHANNELS
std = [0.229] * NUM_INPUT_CHANNELS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        image = tiff.imread(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read TIFF file: {str(e)}"}), 400

    
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] != NUM_INPUT_CHANNELS:
        return jsonify({"error": f"Expected {NUM_INPUT_CHANNELS} channels, but got {image.shape[-1]}."}), 400

    image = np.transpose(image, (2, 0, 1))  # (Bands, Height, Width)
    image = torch.tensor(image, dtype=torch.float32) / 255.0  

    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False).squeeze(0)

    # Normalization 
    if USE_NORMALIZATION:
        normalize = transforms.Normalize(mean=mean, std=std)
        image = normalize(image)

    
    image = image.to(device)

    
    with torch.no_grad():
        output = model(image.unsqueeze(0))  
        output = torch.sigmoid(output).cpu().numpy().squeeze()  

    print("Output min/max before thresholding:", output.min(), output.max())

    mask = (output > 0.5).astype(np.uint8) * 255  

    
    mask_image = Image.fromarray(mask)
    mask_path = os.path.join("static", "mask.png")
    mask_image.save(mask_path)

    return jsonify({"mask_url": url_for('static', filename='mask.png', _external=True)})

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)