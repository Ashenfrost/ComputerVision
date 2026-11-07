import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from collections import Counter
from datetime import datetime

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================================================
# ITERATION 2: klassifikation af UDSKÅRNE skilte
# ---------------------------------------------------------
# Iteration 1 (Machineconvoluted.py) gav hele billedet til
# netværket. Skiltet fyldte kun en lille del, så modellen
# lærte mest af baggrunden. Her bruger vi bounding boxes til
# at skære hvert skilt ud, så netværket kun ser selve skiltet.
# =========================================================

torch.manual_seed(0)  # samme resultat hver gang -> sammenlignelige forsøg

# ---------------------------------------------------------
# 0. Stier og indstillinger
# ---------------------------------------------------------
DATASET_ROOT = Path("cardata")

with open(DATASET_ROOT / "data.yaml") as f:
    data_config = yaml.safe_load(f)

CLASS_NAMES = data_config["names"]
NUM_CLASSES = data_config["nc"]
print(f"Loaded {NUM_CLASSES} classes: {CLASS_NAMES}")

TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
VALID_IMAGES_DIR = DATASET_ROOT / "valid" / "images"
VALID_LABELS_DIR = DATASET_ROOT / "valid" / "labels"

IMG_SIZE = 64        # NYT: et udskåret skilt behøver ikke 418x418
BATCH_SIZE = 128
EPOCHS = 30          # NYT: 10 var for lidt til at træne fra bunden
CROP_PADDING = 0.15  # tag 15% ekstra rundt om boxen, så kanten af skiltet kommer med
MIN_BOX_PIXELS = 8   # spring bokse over der er mindre end 8 px (bare støj)

# Normalisering: flytter pixelværdier fra [0, 1] til ca. [-1, 1]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# Hver kørsel får sin egen mappe, så gamle resultater aldrig overskrives.
# Ret RUN_NAME når du laver en ny iteration (fx "iter3_sqrt_weights").
# Tidsstemplet gør, at selv en genkørsel af samme iteration får en ny mappe.
RUN_NAME = "iter2_crops"
RUN_ID = f"{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
OUT_DIR.mkdir(parents=True)  # fejler hvis mappen findes -> intet overskrives
print(f"Resultater gemmes i {OUT_DIR}")


# ---------------------------------------------------------
# 1. Dataset — ét sample pr. bounding box (ikke pr. billede)
# ---------------------------------------------------------
class CroppedSignDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.transform = transform
        self.samples = []  # (image_path, class_id, (x1, y1, x2, y2) normaliseret)

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
            stem = label_file[:-4]
            image_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(images_dir, stem + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                continue

            with open(os.path.join(labels_dir, label_file)) as f:
                lines = [l.strip() for l in f if l.strip()]

            # YOLO-format: class cx cy w h  (alle værdier 0-1, relativt til billedet)
            # FORSKEL fra iteration 1: vi beholder ALLE bokse, ikke kun den største
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # udvid boksen med padding, og hold den inden for billedet (0-1)
                w_pad, h_pad = w * (1 + CROP_PADDING), h * (1 + CROP_PADDING)
                x1 = max(0.0, cx - w_pad / 2)
                y1 = max(0.0, cy - h_pad / 2)
                x2 = min(1.0, cx + w_pad / 2)
                y2 = min(1.0, cy + h_pad / 2)
                self.samples.append((image_path, class_id, (x1, y1, x2, y2)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_id, (x1, y1, x2, y2) = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        # fra normaliserede koordinater (0-1) til pixels
        box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
        crop = image.crop(box)

        if self.transform:
            crop = self.transform(crop)
        return crop, class_id

    def class_counts(self):
        return Counter(class_id for _, class_id, _ in self.samples)


def remove_tiny_boxes(dataset, images_dir):
    """Fjerner bokse der er under MIN_BOX_PIXELS i bredde eller højde."""
    size_cache = {}
    kept = []
    for image_path, class_id, (x1, y1, x2, y2) in dataset.samples:
        if image_path not in size_cache:
            with Image.open(image_path) as im:
                size_cache[image_path] = im.size
        W, H = size_cache[image_path]
        if (x2 - x1) * W >= MIN_BOX_PIXELS and (y2 - y1) * H >= MIN_BOX_PIXELS:
            kept.append((image_path, class_id, (x1, y1, x2, y2)))
    removed = len(dataset.samples) - len(kept)
    dataset.samples = kept
    print(f"Removed {removed} tiny boxes from {images_dir}")


# ---------------------------------------------------------
# 2. Model — samme arkitektur som iteration 1
# ---------------------------------------------------------
# Vi holder netværket ens, så forskellen i resultat kun skyldes
# data/træning. 64x64 input: 64 -> 32 -> 16 -> 8 -> 4 efter 4 pooling-lag.
class Net(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)          # 64 -> 32

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)          # 32 -> 16

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)          # 16 -> 8

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)          # 8 -> 4
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------
# 3. Transforms — NYT: augmentation på træningsdata
# ---------------------------------------------------------
# Augmentation = små tilfældige ændringer hver gang et billede bruges,
# så modellen ikke kan lære træningsbillederne udenad.
# BEVIDST FRAVALGT:
#   - RandomHorizontalFlip: spejlvender tallene ("30" bliver til noget andet)
#   - Hue i ColorJitter: rød/grøn er selve forskellen på Red/Green Light
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])

# valideringsdata må IKKE augmenteres -- vi vil måle på de rigtige billeder
test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])

training_data = CroppedSignDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=train_transform)
test_data = CroppedSignDataset(VALID_IMAGES_DIR, VALID_LABELS_DIR, transform=test_transform)
remove_tiny_boxes(training_data, TRAIN_IMAGES_DIR)
remove_tiny_boxes(test_data, VALID_IMAGES_DIR)

print(f"Train samples (crops): {len(training_data)}, Test samples (crops): {len(test_data)}")

train_counts = training_data.class_counts()
print("\nTrain crops per class:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<16} {train_counts.get(i, 0)}")

# num_workers=0: på Windows kræver flere workers en if __name__ == "__main__" guard
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)


# ---------------------------------------------------------
# 4. Model, optimizer, loss
# ---------------------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"\nUsing {device} device")

model = Net(num_classes=NUM_CLASSES).to(device)
model(torch.rand((1, 3, IMG_SIZE, IMG_SIZE)).to(device))  # initialiser LazyLinear
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# NYT: class weights mod ubalance.
# Sjældne klasser får større vægt, så en fejl på "Speed Limit 10"
# koster mere end en fejl på en klasse med masser af eksempler.
# vægt = total / (antal_klasser * antal_i_klassen)
total = sum(train_counts.values())
class_weights = torch.tensor(
    [total / (NUM_CLASSES * max(train_counts.get(i, 0), 1)) for i in range(NUM_CLASSES)],
    dtype=torch.float32,
).to(device)
print("\nClass weights:", [round(w, 2) for w in class_weights.tolist()])
loss_fn = nn.NLLLoss(weight=class_weights)

# NYT: sænk learning rate hver 10. epoch i stedet for hver 5.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

writer = SummaryWriter(f"runs/{RUN_ID}")  # egen mappe pr. kørsel, ellers blandes kurverne


# ---------------------------------------------------------
# 5. Train / test loops (uændret fra iteration 1)
# ---------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
            global_step = epoch * len(dataloader) + batch
            writer.add_scalar("Loss/train", loss_val, global_step)


def test(dataloader, model, loss_fn, epoch, writer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", correct, epoch)
    return correct


# ---------------------------------------------------------
# 6. Træning — NYT: gem den bedste model, ikke bare den sidste
# ---------------------------------------------------------
best_acc = 0.0
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t, writer)
    acc = test(test_dataloader, model, loss_fn, t, writer)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), OUT_DIR / "model.pth")
        print(f"New best model saved ({100*acc:.1f}%)")
    scheduler.step()
    writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], t)

writer.close()
print(f"Done! Best accuracy: {100*best_acc:.1f}%")

# indlæs den bedste model til evaluering
model.load_state_dict(torch.load(OUT_DIR / "model.pth", map_location=device))
model.eval()


# ---------------------------------------------------------
# 7. Confusion matrix + per-klasse rapport
# ---------------------------------------------------------
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        pred = model(X).argmax(1).cpu()
        all_preds.extend(pred.tolist())
        all_labels.extend(y.tolist())

# NYT: precision/recall pr. klasse -- god til rapporten, fordi
# samlet accuracy kan skjule klasser modellen aldrig rammer
report = classification_report(all_labels, all_preds, labels=list(range(NUM_CLASSES)),
                               target_names=CLASS_NAMES, zero_division=0)
print(report)
with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"{RUN_ID}\nBest accuracy: {100*best_acc:.1f}%\n\n{report}")

cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {RUN_ID}")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png")
print(f"Saved confusion matrix to {OUT_DIR / 'confusion_matrix.png'}")


# ---------------------------------------------------------
# 8. Grad-CAM: kigger modellen på tallet?
# ---------------------------------------------------------
target_layers = [model.conv4]
cam = GradCAM(model=model, target_layers=target_layers)

# vælg eksempler spredt ud over testsættet, så vi ikke kun ser én klasse
num_examples = 8
indices = np.linspace(0, len(test_data) - 1, num_examples, dtype=int)
fig, axes = plt.subplots(2, num_examples // 2, figsize=(3 * num_examples // 2, 6.5))
for ax, i in zip(axes.flat, indices):
    img_tensor, label = test_data[i]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]  # None = forklar den forudsagte klasse

    # fortryd normaliseringen, så billedet kan vises (show_cam_on_image vil have 0-1)
    rgb_img = img_tensor.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
    rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        pred_idx = model(input_tensor)[0].argmax(0).item()

    ax.imshow(visualization)
    ax.set_title(f"Actual: {CLASS_NAMES[label]}\nPred: {CLASS_NAMES[pred_idx]}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "gradcam_examples.png")
print(f"Saved Grad-CAM visualizations to {OUT_DIR / 'gradcam_examples.png'}")
