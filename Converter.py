import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --- CONFIG ---
IMG_SIZE = 128  # Resize for speed
LATENT_DIM = 64
THRESHOLD_PERCENTILE = 95  # How strict we are
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Shrink the image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        # Decoder: Rebuild the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_and_detect(input_dir):
    # 1. PREPARE DATA
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')]

    # Heuristic: We assume MOST images are good. We train on all of them.
    # The corrupted ones are "outliers" and won't be learned well.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # 2. TRAIN (Quick & Dirty)
    model = SimpleAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("🧠 Training Anomaly Detector on your dataset...")
    model.train()
    for epoch in range(50):  # Quick training
        for img_path in all_files:
            img = cv2.imread(img_path)
            if img is None: continue
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            # Forward
            recon = model(tensor)
            loss = criterion(recon, tensor)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 3. DETECT (Evaluate)
    print("\n🔎 Scanning for Anomalies...")
    model.eval()
    errors = []
    results = []

    for img_path in all_files:
        img = cv2.imread(img_path)
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            recon = model(tensor)

        # Calculate Difference (MSE) per image
        diff = torch.mean((recon - tensor) ** 2).item()
        errors.append(diff)
        results.append((img_path, diff))

    # 4. DECIDE
    # Anything with error > 95th percentile of normal is an anomaly
    threshold = np.percentile(errors, THRESHOLD_PERCENTILE)

    print(f"📊 Anomaly Threshold: {threshold:.5f}")

    for path, err in results:
        status = "✅ PASS"
        if err > threshold:
            status = "🚩 REPAIR (High Reconstruction Error)"
            print(f"{status} | {os.path.basename(path)} | Error: {err:.5f}")


if __name__ == "__main__":
    # Point this to your ./train folder
    train_and_detect("./output_processed/chair/train")