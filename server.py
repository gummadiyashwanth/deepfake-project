from flask import Flask, render_template, request, json
from werkzeug.utils import secure_filename

import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from torch.autograd import Variable
from torch import nn
from torchvision import models
import warnings

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # Load a pre-trained model
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])

        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

# Convert tensor image to OpenCV format
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image * 255)
    return image

# Prediction function
def predict(model, img, path='./'):
    fmap, logits = model(img.to())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('Confidence of prediction:', confidence)
    return [int(prediction.item()), confidence]

# Dataset class for video processing
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = self.detect_faces(frame)
            try:
                x, y, w, h = faces[0]
                frame = frame[y:y + h, x:x + w]  # Crop to detected face
            except:
                pass  # If no face detected, continue with full frame

            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    # Extract frames from the video
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, frame = vidObj.read()
            if success:
                yield frame

    # Use OpenCV for face detection
    def detect_faces(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

# Fake video detection function
def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    path_to_videos = [videoPath]
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)

    model = Model(2)
    path_to_model = 'model/my_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()

    for i in range(len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model, video_dataset[i], './')
        print("REAL" if prediction[0] == 1 else "FAKE")
    return prediction

# Flask Routes
@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        video = request.files['video']
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        prediction = detectFakeVideo(video_path)
        print(prediction)
        output = "FAKE" if prediction[0] == 0 else "REAL"
        confidence = prediction[1]
        data = {'output': output, 'confidence': confidence}

        os.remove(video_path)
        return render_template('index.html', data=json.dumps(data))

# Run the Flask app
app.run(port=3000)
