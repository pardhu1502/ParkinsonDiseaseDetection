import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from torchvision import models
from torchvision.models import ResNet18_Weights

# =====================
# CONFIG
# =====================
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pd_path = "../PD Patients"
nonpd_path = "../Non PD Patients"

# =====================
# SEED
# =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

# =====================
# DATASET
# =====================
class ParkinsonDataset(Dataset):
    def __init__(self, subject_list, max_samples_per_subject=40):
        self.samples = []

        for subject, base_path, label in subject_list:
            path = os.path.join(base_path, subject)

            mri_path = os.path.join(path, "1.MRI")
            dat_path = os.path.join(path, "0.DAT")

            if not os.path.exists(mri_path) or not os.path.exists(dat_path):
                continue

            # MRI
            mri_files = sorted(os.listdir(mri_path))
            mri_files = [os.path.join(mri_path, f) for f in mri_files]

            if len(mri_files) < 3:
                continue

            # Reduce redundancy
            mri_files = mri_files[::15]

            triplets = []
            for i in range(1, len(mri_files)-1):
                triplets.append([mri_files[i-1], mri_files[i], mri_files[i+1]])

            # DAT
            dat_images = []
            for folder in os.listdir(dat_path):
                subfolder = os.path.join(dat_path, folder)
                if os.path.isdir(subfolder):
                    for img in os.listdir(subfolder):
                        dat_images.append(os.path.join(subfolder, img))

            if len(dat_images) == 0:
                continue

            dat_images = dat_images[:10]  # limit DAT

            random.shuffle(dat_images)

            # 🔥 1-to-1 mapping (fix duplicates)
            count = 0
            for i, triplet in enumerate(triplets):
                dat = dat_images[i % len(dat_images)]
                self.samples.append((triplet, dat, label))
                count += 1
                if count >= max_samples_per_subject:
                    break

        # =====================
        # BALANCE DATASET
        # =====================
        random.shuffle(self.samples)

        healthy = [s for s in self.samples if s[2] == 0]
        pd = [s for s in self.samples if s[2] == 1]

        min_len = min(len(healthy), len(pd))
        self.samples = healthy[:min_len] + pd[:min_len]

        random.shuffle(self.samples)

        print("Balanced Dataset size:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def safe_read(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    def __getitem__(self, idx):
        triplet, dat_path, label = self.samples[idx]

        imgs = []

        for p in triplet:
            img = self.safe_read(p)
            img = torch.tensor(img).permute(2,0,1).float()/255.0
            img = (img - 0.5) / 0.5  # normalize
            imgs.append(img)

        dat = self.safe_read(dat_path)
        dat = torch.tensor(dat).permute(2,0,1).float()/255.0
        dat = (dat - 0.5) / 0.5

        imgs.append(dat)

        return torch.stack(imgs), torch.tensor(label).float()

# =====================
# LOAD DATA
# =====================
def get_subjects(base_path, label):
    return [(s, base_path, label) for s in os.listdir(base_path)]

all_subjects = get_subjects(pd_path, 1) + get_subjects(nonpd_path, 0)

train_subjects, test_subjects = train_test_split(
    all_subjects, test_size=0.2, random_state=42
)

train_dataset = ParkinsonDataset(train_subjects)
test_dataset = ParkinsonDataset(test_subjects)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# MODEL
# =====================
class ParkinsonModel(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        self.gru = nn.GRU(512, 128, batch_first=True)
        self.fc_dat = nn.Linear(512, 128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch = x.size(0)

        mri_feats = []
        for i in range(3):
            f = self.cnn(x[:, i]).view(batch, -1)
            mri_feats.append(f)

        mri_seq = torch.stack(mri_feats, dim=1)
        _, h = self.gru(mri_seq)
        mri_out = h.squeeze(0)

        dat = self.cnn(x[:, 3]).view(batch, -1)
        dat_out = F.relu(self.fc_dat(dat))

        # 🔥 Increase DAT importance
        x = torch.cat([mri_out * 0.5, dat_out * 1.5], dim=1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# =====================
# TRAIN FUNCTION
# =====================
def train_model(model):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    return model

# =====================
# TRAIN
# =====================
model = train_model(ParkinsonModel())

# =====================
# EVALUATE
# =====================
model.eval()
y_true, y_probs = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()

        y_probs.extend(probs)
        y_true.extend(labels.numpy())

# 🔥 Better threshold
THRESHOLD = 0.75
y_pred = [1 if p > THRESHOLD else 0 for p in y_probs]

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# =====================
# SAVE MODEL
# =====================
torch.save(model.state_dict(), "parkinson_final.pth")
print("\nModel saved successfully")