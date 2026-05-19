import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

from gradcam import GradCAM, overlay_cam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        img = x[:, 3]
        f = self.pool(self.cnn(img)).view(x.size(0), -1)
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
        img = x[:, 3]
        f = self.cnn(img).view(x.size(0), -1)
        return self.fc(f)


# =====================
# LOAD MODELS
# =====================
model1 = Model1().to(device)
model2 = Model2().to(device)
model3 = Model3().to(device)

model1.load_state_dict(torch.load("model7.pth", map_location=device))
model2.load_state_dict(torch.load("model8.pth", map_location=device))
model3.load_state_dict(torch.load("model9.pth", map_location=device))

models = [model1, model2, model3]

# =====================
# HELPERS
# =====================
def compute_eds(contour):
    if contour is None or len(contour) < 5:
        return 0.0

    ellipse = cv2.fitEllipse(contour)
    (_, _), (MA, ma), _ = ellipse

    if ma == 0:
        return 0.0

    return float(abs(MA - ma) / max(MA, ma))


def extract_brain(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img, None

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    brain = cv2.bitwise_and(img, img, mask=mask)
    return brain, largest 


def tta_predict(model, x):
    preds = []
    for _ in range(5):
        noisy = x + torch.randn_like(x) * 0.01
        preds.append(torch.sigmoid(model(noisy)).item())
    return np.mean(preds)


# =====================
# MAIN FUNCTION (USED BY FLASK)
# =====================
def run_inference(image_path):

    img = cv2.imread(image_path)

    brain, contour = extract_brain(img)
    eds = compute_eds(contour)

    tensor_img = cv2.resize(img, (224,224))
    tensor_img = torch.tensor(tensor_img).permute(2,0,1).float()/255

    x = torch.stack([tensor_img]*4).unsqueeze(0).to(device)

    weights = [0.5, 0.2, 0.3]
    probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            probs.append(tta_predict(model, x))

    model_score = sum(w*p for w,p in zip(weights, probs))

    # Decision
    if model_score > 0.85:
        pred="PD (Model confident)"
    elif model_score > 0.80 and model_score < 0.85:
        if eds > 0.18:
            pred = "PD (Confirmed by EDS)"
        else:
            # pred = "PD (Model confident)"
            pred="Healthy"
    elif model_score < 0.80:
        pred = "Healthy"
    # else:
    #     pred = "Uncertain"

    # GradCAM
    gradcam = GradCAM(model1, model1.cnn[-2])
    cam = gradcam.generate(x)

    img_show = x[0, 3].cpu().numpy().transpose(1,2,0)
    img_show = (img_show * 255).astype(np.uint8)

    overlay = overlay_cam(img_show, cam)

    output_path = "static/results/result.jpg"
    cv2.imwrite(output_path, overlay)

    return pred, model_score, eds, output_path