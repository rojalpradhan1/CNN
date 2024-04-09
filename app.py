from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision.transforms import transforms
import numpy as np
import torch.nn as nn

# Initialize Flask application
app = Flask(__name__)

# Define the model architecture class
class KD32Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 8x8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)  # Assuming 6 emotions
        )

    def forward(self, xb):
        return self.network(xb)

# Load the model
model = KD32Model()
model.load_state_dict(torch.load('C://Users//ritik//Downloads//rojal_dl//models//from_dataLoader.pt'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define emotion labels
emotions = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['image']
    # Open and preprocess the image
    image = Image.open(file.stream).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    # Get the predicted emotion label
    predicted_emotion = emotions[prediction]
    # Return the predicted emotion
    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)