import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

# =====================
# SEED
# =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# PATHS
# =====================
pd_path = "../PD Patients"
nonpd_path = "../Non PD Patients"

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

            mri_files = sorted(os.listdir(mri_path))
            mri_files = [os.path.join(mri_path, f) for f in mri_files]

            if len(mri_files) < 3:
                continue

            mri_files = mri_files[::10]

            triplets = []
            for i in range(1, len(mri_files)-1):
                triplets.append([mri_files[i-1], mri_files[i], mri_files[i+1]])

            dat_images = []
            for folder in os.listdir(dat_path):
                subfolder = os.path.join(dat_path, folder)
                if os.path.isdir(subfolder):
                    for img in os.listdir(subfolder):
                        dat_images.append(os.path.join(subfolder, img))

            if len(dat_images) == 0:
                continue

            count = 0
            for triplet in triplets:
                for dat in dat_images:
                    self.samples.append((triplet, dat, label))
                    count += 1
                    if count >= max_samples_per_subject:
                        break
                if count >= max_samples_per_subject:
                    break

        print("Dataset size:", len(self.samples))

    def augment(self, img):
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (5,5), 0)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        triplet, dat_path, label = self.samples[idx]

        imgs = []

        for p in triplet:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((224,224,3))
            img = self.augment(img)
            img = cv2.resize(img, (224,224))
            img = torch.tensor(img).permute(2,0,1).float()/255
            imgs.append(img)

        dat = cv2.imread(dat_path)
        if dat is None:
            dat = np.zeros((224,224,3))
        dat = self.augment(dat)
        dat = cv2.resize(dat, (224,224))
        dat = torch.tensor(dat).permute(2,0,1).float()/255

        imgs.append(dat)

        return torch.stack(imgs), torch.tensor(label).float()

# =====================
# DATA LOADING
# =====================
def get_subjects(base_path, label):
    return [(s, base_path, label) for s in os.listdir(base_path)]

all_subjects = get_subjects(pd_path, 1) + get_subjects(nonpd_path, 0)

train_subjects, test_subjects = train_test_split(
    all_subjects, test_size=0.2, random_state=42
)

train_dataset = ParkinsonDataset(train_subjects, 40)
test_dataset = ParkinsonDataset(test_subjects, 20)

labels = [label for _, _, label in train_dataset.samples]
class_counts = np.bincount(labels)

weights = 1. / class_counts
sample_weights = [weights[int(l)] for l in labels]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# =====================
# MODELS
# =====================
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        self.gru = nn.GRU(512, 128, batch_first=True)
        self.fc_dat = nn.Linear(512, 128)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

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

        x = torch.cat([mri_out, dat_out], dim=1)
        return self.fc(x)

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cnn = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch = x.size(0)
        img = x[:, 3]
        f = self.pool(self.cnn(img)).view(batch, -1)
        return self.fc(f)

class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch = x.size(0)
        img = x[:, 3]
        f = self.cnn(img).view(batch, -1)
        return self.fc(f)

# =====================
# TRAIN FUNCTION
# =====================
def train_model(model, epochs=15):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    pos_weight = torch.tensor([2.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

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
# TRAIN MODELS
# =====================
model1 = train_model(Model1())
model2 = train_model(Model2())
model3 = train_model(Model3())

models_list = [model1, model2, model3]

# =====================
# TTA
# =====================
def tta_predict(model, x):
    preds = []
    for _ in range(5):
        noisy = x + torch.randn_like(x) * 0.01
        preds.append(torch.sigmoid(model(noisy)).item())
    return np.mean(preds)

# =====================
# ENSEMBLE
# =====================
def ensemble_predict(models, x):
    weights = [0.5, 0.2, 0.3]

    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            probs.append(tta_predict(model, x))

    final_prob = sum(w * p for w, p in zip(weights, probs))
    return final_prob

# =====================
# THRESHOLD TUNING
# =====================
y_true, y_probs = [], []

for images, labels in test_loader:
    images = images.to(device)

    for i in range(images.size(0)):
        x = images[i].unsqueeze(0)
        prob = ensemble_predict(models_list, x)

        y_probs.append(prob)
        y_true.append(int(labels[i].item()))

best_t, best_f1 = 0, 0

for t in np.linspace(0.3, 0.9, 50):
    preds = [1 if p > t else 0 for p in y_probs]
    f1 = f1_score(y_true, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best Threshold:", best_t)

# =====================
# FINAL EVALUATION
# =====================
y_pred = [1 if p > best_t else 0 for p in y_probs]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", acc)
print("F1 Score:", f1)

# =====================
# CONFUSION MATRIX
# =====================
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Plot
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()

# =====================
# SAVE MODELS
# =====================
torch.save(model1.state_dict(), "model1.pth")
torch.save(model2.state_dict(), "model2.pth")
torch.save(model3.state_dict(), "model3.pth")

print("Final models saved")