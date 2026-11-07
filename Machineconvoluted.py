import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------------------------------------------
# 0. Paths — loaded from data.yaml, only edit DATASET_ROOT
# ---------------------------------------------------------
DATASET_ROOT = Path("cardata")  # <-- change this: folder containing data.yaml

with open(DATASET_ROOT / "data.yaml") as f:
    data_config = yaml.safe_load(f)

CLASS_NAMES = data_config["names"]
NUM_CLASSES = data_config["nc"]
print(f"Loaded {NUM_CLASSES} classes: {CLASS_NAMES}")

# data.yaml paths are relative to the yaml file's own location and use
# "val" as the key, but the actual folder on disk may be named "valid" --
# adjust these two lines if your folder names differ
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
VALID_IMAGES_DIR = DATASET_ROOT / "valid" / "images"
VALID_LABELS_DIR = DATASET_ROOT / "valid" / "labels"

IMG_SIZE = 418
BATCH_SIZE = 64
EPOCHS = 10

# Hver kørsel får sin egen mappe, så gamle resultater aldrig overskrives.
RUN_NAME = "iter1_fullimage"
RUN_ID = f"{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
OUT_DIR.mkdir(parents=True)
print(f"Results will be saved in {OUT_DIR}")


# ---------------------------------------------------------
# 1. Dataset — full images, labeled by the largest box's class
# ---------------------------------------------------------
class FullImageSignDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.samples = []  # (image_path, class_id)

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
            if not lines:
                continue  # skip images with no labeled objects

            # pick the class of the largest bounding box (w * h) as the
            # "main subject" label for the whole image
            best_line = max(lines, key=lambda l: float(l.split()[3]) * float(l.split()[4]))
            class_id = int(best_line.split()[0])
            self.samples.append((image_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_id = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_id


# ---------------------------------------------------------
# 2. Model
# ---------------------------------------------------------
# Modernized vs. the original recipe:
# - BatchNorm2d after each conv: stabilizes/speeds up training
# - An extra conv+pool block: at 418x418 input, only 1 pooling step
#   (the original recipe's design) leaves a huge, slow-to-train,
#   overfit-prone flatten. Pooling after every block shrinks the
#   spatial size much faster: 418 -> 209 -> 104 -> 52 -> 26.
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
        self.fc1 = nn.LazyLinear(256)   # auto-sizes on first forward pass
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)          # 418 -> 209

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)          # 209 -> 104

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)          # 104 -> 52

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)          # 52 -> 26
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------
# 3. Build datasets and dataloaders
# ---------------------------------------------------------
transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

training_data = FullImageSignDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)
test_data = FullImageSignDataset(VALID_IMAGES_DIR, VALID_LABELS_DIR, transform=transform)

print(f"Train samples: {len(training_data)}, Test samples: {len(test_data)}")

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# ---------------------------------------------------------
# 4. Build model, optimizer, loss
# ---------------------------------------------------------
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = Net(num_classes=NUM_CLASSES).to(device)

# trigger LazyLinear's shape inference before printing/training
dummy = torch.rand((1, 3, IMG_SIZE, IMG_SIZE)).to(device)
model(dummy)
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.NLLLoss()  # pairs with log_softmax output in Net.forward

# decays the learning rate by 10x every 5 epochs -- takes bigger steps
# early in training, smaller/more careful steps later on
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# logs scalars to ./runs/<RUN_ID> -- view with:
#   tensorboard --logdir=runs
writer = SummaryWriter(f"runs/{RUN_ID}")


# ---------------------------------------------------------
# 5. Train / test loops
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


# ---------------------------------------------------------
# 6. Run training
# ---------------------------------------------------------
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t, writer)
    test(test_dataloader, model, loss_fn, t, writer)
    scheduler.step()
    writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], t)
    print(f"Learning rate now: {scheduler.get_last_lr()[0]}\n")

writer.close()
print("Done!")
torch.save(model.state_dict(), OUT_DIR / "model.pth")
print(f"Saved model to {OUT_DIR / 'model.pth'}")

# ---------------------------------------------------------
# 7. Try a single prediction with a human-readable class name
# ---------------------------------------------------------
model.eval()
x, y = test_data[0]
with torch.no_grad():
    x = x.unsqueeze(0).to(device)
    pred = model(x)
    predicted_idx = pred[0].argmax(0).item()
    print(f'Predicted: "{CLASS_NAMES[predicted_idx]}", Actual: "{CLASS_NAMES[y]}"')


# ---------------------------------------------------------
# 8. Confusion matrix on the full test set
# ---------------------------------------------------------
# Shows which classes get mixed up with which -- important here since
# some classes (e.g. "Speed Limit 30") have far fewer training examples
# than others (e.g. "Red Light"), so overall accuracy alone can hide
# classes the model rarely predicts correctly.
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        pred = model(X).argmax(1).cpu()
        all_preds.extend(pred.tolist())
        all_labels.extend(y.tolist())

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
# 9. Grad-CAM: visualize where the model is "looking"
# ---------------------------------------------------------
# Overlays a heatmap on the input image showing which pixels most
# influenced the prediction. Useful here to check whether the model
# is actually focusing on the sign itself, or cheating using
# background/context, given several images contain multiple signs.
target_layers = [model.conv4]  # last conv layer, most semantically meaningful
cam = GradCAM(model=model, target_layers=target_layers)

num_examples = 4
fig, axes = plt.subplots(1, num_examples, figsize=(4 * num_examples, 4))
for i in range(num_examples):
    img_tensor, label = test_data[i]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = img_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC for display
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        pred_idx = model(input_tensor)[0].argmax(0).item()

    axes[i].imshow(visualization)
    axes[i].set_title(f"Actual: {CLASS_NAMES[label]}\nPred: {CLASS_NAMES[pred_idx]}", fontsize=9)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "gradcam_examples.png")
print(f"Saved Grad-CAM visualizations to {OUT_DIR / 'gradcam_examples.png'}")