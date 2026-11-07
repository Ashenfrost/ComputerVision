import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# =========================================================
# Fælles kode til test-scripts (negative_test.py, robustness_test.py)
# ---------------------------------------------------------
# De tre detektionsmetoder + hjælpefunktioner til at bygge
# testsæt, så alle tests bruger præcis samme pipelines.
# =========================================================

CNN_WEIGHTS = Path("results/iter2_crops_20261102_152508/model.pth")
YOLO_WEIGHTS = Path("results/yolov8n_20261102_163040/weights/best.pt")
DATA_DIR = Path("cardata")  # indeholder train/, valid/ og test/
STRAWBERRY_DIR = Path("strawberry")

IMG_SIZE = 416           # alle billeder bringes til samme størrelse som cardata
IOU_MATCH = 0.5          # en detektion "rammer" et skilt hvis IoU >= 0.5
YOLO_CONF = 0.25         # Ultralytics' standard-konfidensgrænse

with open("cardata/data.yaml") as f:
    CLASS_NAMES = yaml.safe_load(f)["names"]
RED_CLASSES = set(range(2, 15))  # alle hastighedsskilte + Stop har rød kant (ikke lys)


# ---------------------------------------------------------
# 1. Klassisk pipeline (fra cannytest.py afsnit 4 + 6)
# ---------------------------------------------------------
# cannytest.py brugte medianBlur(13) og minRadius=30, tunet på ét
# billede (30.2.jpg). I testsættet er mange skilte langt mindre end
# det, så blur og minimumsradius er sænket.
LOWER_RED1, UPPER_RED1 = np.array([0, 100, 70]), np.array([8, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([172, 100, 70]), np.array([180, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)

# Pipelinen bygges op i tre trin, så effekten af hvert filter kan måles:
#   A: naiv        -- løs Hough (mange cirkler oven i hinanden)
#   B: strammere   -- større minDist + højere param2 (kræver mere "bevis")
#   C: ring-tjek   -- et skilt har RØD RING og HVID MIDTE; et jordbær er rødt hele vejen
HOUGH_PARAMS = {
    "A": dict(minDist=20, param2=15),
    "B": dict(minDist=40, param2=30),
    "C": dict(minDist=40, param2=30),
}
RING_MIN_RED = 0.5     # mindst 50% af omkredsen skal være rød
CENTER_MAX_RED = 0.3   # højst 30% af midten må være rød


def red_mask(img, steps=None):
    """steps: giv en tom dict for at få mellemresultatet efter hvert filter med
    (bruges af pipeline_steps.py til figurer). Påvirker ikke resultatet."""
    blurred = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1) | cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, KERNEL)   # luk huller i den røde ring
    mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, KERNEL)   # fjern små røde prikker
    if steps is not None:
        steps.update(median=blurred, hsv_mask=raw, close=closed, open=mask)
    return mask


def looks_like_sign(mask, x, y, r):
    """Ring-tjek: rød omkreds og ikke-rød midte."""
    H, W = mask.shape
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring_hits = 0
    for a in angles:
        # tæl punktet som rødt hvis ringen er rød et sted mellem 0.75r og 1.0r
        for rr in (0.75 * r, 0.875 * r, r):
            px, py = int(x + rr * np.cos(a)), int(y + rr * np.sin(a))
            if 0 <= px < W and 0 <= py < H and mask[py, px]:
                ring_hits += 1
                break
    center = np.zeros_like(mask)
    cv2.circle(center, (int(x), int(y)), max(1, int(0.5 * r)), 255, -1)
    center_red = (mask[center > 0] > 0).mean()
    return ring_hits / len(angles) >= RING_MIN_RED and center_red <= CENTER_MAX_RED


def classical_detect(img, stage="C", min_dist=None, param2=None, ring_check=None, steps=None):
    """Returnerer liste af bokse (x1, y1, x2, y2) i pixels.

    stage vælger et af de faste trin A/B/C. min_dist, param2 og
    ring_check kan overskrive trinnets værdier (bruges i parametersweep).
    steps: giv en tom dict for at få alle mellemresultater med (se red_mask),
    plus "gaussian", "circles" (alle Hough-cirkler) og "kept" (efter ring-tjek).
    """
    params = dict(HOUGH_PARAMS[stage])
    if min_dist is not None:
        params["minDist"] = min_dist
    if param2 is not None:
        params["param2"] = param2
    if ring_check is None:
        ring_check = stage == "C"

    mask = red_mask(img, steps)
    smooth = cv2.GaussianBlur(mask, (5, 5), 0)  # bløde kanter -> bedre gradienter til Hough
    circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=1.2, param1=100,
                               minRadius=6, maxRadius=150, **params)
    boxes, kept = [], []
    if circles is not None:
        for x, y, r in circles[0]:
            if ring_check and not looks_like_sign(mask, x, y, r):
                continue
            kept.append((x, y, r))
            boxes.append((x - r, y - r, x + r, y + r))
    if steps is not None:
        steps.update(gaussian=smooth, circles=[] if circles is None else list(circles[0]), kept=kept)
    return boxes


# ---------------------------------------------------------
# 2. CNN fra iteration 2 (kopi af Net fra Machineconvoluted_crops.py,
#    fordi import af det script ville starte en ny træning)
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
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
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = self.dropout1(x)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        return F.log_softmax(self.fc2(self.dropout2(x)), dim=1)


_models = {}  # modellerne indlæses først når de bruges, og kun én gang


def get_cnn(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = ("cnn", device)
    if key not in _models:
        cnn = Net(len(CLASS_NAMES)).to(device)
        cnn(torch.rand(1, 3, 64, 64).to(device))  # initialiser LazyLinear
        cnn.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))
        cnn.eval()
        _models[key] = cnn
    return _models[key]


def cnn_classify(img, box, device=None):
    """Skærer boksen ud (+15% padding som i træningen) og returnerer (klasse, sikkerhed)."""
    cnn = get_cnn(device)
    dev = next(cnn.parameters()).device
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * 0.075, (y2 - y1) * 0.075
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, 0.0
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0
    t = torch.from_numpy((crop - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs = cnn(t).exp()[0]
    conf, cls = probs.max(0)
    return int(cls), float(conf)


# ---------------------------------------------------------
# 3. YOLO
# ---------------------------------------------------------
def get_yolo(weights=YOLO_WEIGHTS):
    key = ("yolo", str(weights))
    if key not in _models:
        from ultralytics import YOLO
        _models[key] = YOLO(weights)
    return _models[key]


def yolo_detect(img, device=None, weights=YOLO_WEIGHTS, imgsz=IMG_SIZE):
    r = get_yolo(weights).predict(img, imgsz=imgsz, conf=YOLO_CONF, verbose=False, device=device)[0]
    return [(tuple(b), int(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]


# ---------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, dets):
    """Parrer sande skilte med detektioner (IoU >= IOU_MATCH, hver detektion bruges én gang).

    Returnerer (liste med detektionens klasse eller False pr. sandt skilt, antal falske bokse).
    En klasse på None betyder fundet, men uden klasse (klassisk pipeline).
    """
    used, result = set(), []
    for gc, gb in gt:
        best_j, best_iou = None, IOU_MATCH
        for j, (db, dc) in enumerate(dets):
            if j not in used and iou(gb, db) >= best_iou:
                best_j, best_iou = j, iou(gb, db)
        if best_j is None:
            result.append(False)
        else:
            used.add(best_j)
            result.append(dets[best_j][1])
    return result, len(dets) - len(used)


def read_gt(label_path, W, H):
    boxes = []
    for line in open(label_path):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        boxes.append((c, ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
    return boxes


def find_image(stem, split="test"):
    for ext in (".jpg", ".jpeg", ".png"):
        p = DATA_DIR / split / "images" / (stem + ext)
        if p.exists():
            return p
    return None


def resize_long_side(img, size=IMG_SIZE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)))


def source_of(filename):
    """Datasættet er samlet fra flere kilder -- kan ses på filnavnet (samme som label_check.py)."""
    if re.match(r"\d{5}_\d{5}_", filename):
        return "GTSRB-udsnit"
    if filename.startswith("FisheyeCamera"):
        return "Simulator (fisheye)"
    if filename.startswith("road"):
        return "road*.png"
    return "Vejfoto (000xxx)"


# ---------------------------------------------------------
# Testsæt. Bruger det globale random-modul, så kalderen styrer
# seed og rækkefølge (samme seed -> samme billeder).
# ---------------------------------------------------------
def label_files(split="test"):
    return sorted((DATA_DIR / split / "labels").glob("*.txt"))


def build_positives(n, split="test"):
    """n tilfældige billeder med mindst ét skilt: [(navn, billede, [(klasse, boks)])]."""
    files = label_files(split)
    positives = []
    for lf in random.sample(files, len(files)):
        img_path = find_image(lf.stem, split)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        gt = read_gt(lf, img.shape[1], img.shape[0])
        if gt:
            positives.append((lf.stem, img, gt))
        if len(positives) == n:
            break
    return positives


def build_strawberries():
    """Jordbær -- røde og runde, men ikke skilte."""
    paths = sorted(p for p in STRAWBERRY_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [(p.stem, resize_long_side(cv2.imread(str(p)))) for p in paths]


def build_road_negatives(n, split="test"):
    """Udsnit af vejbilleder, valgt så de IKKE overlapper nogen skilt-boks.
    GTSRB-nærbilleder springes over (skiltet fylder hele billedet)."""
    candidates = [lf for lf in label_files(split) if not lf.stem[:12].replace("_", "").isdigit()]
    negatives, tries = [], 0
    while len(negatives) < n and tries < 20000:
        tries += 1
        lf = random.choice(candidates)
        img_path = find_image(lf.stem, split)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        gt = read_gt(lf, W, H)
        size = random.randint(int(0.35 * W), int(0.6 * W))
        x, y = random.randint(0, W - size), random.randint(0, H - size)
        crop_box = (x - 10, y - 10, x + size + 10, y + size + 10)  # 10 px sikkerhedsmargin
        if any(not (b[2] < crop_box[0] or b[0] > crop_box[2] or b[3] < crop_box[1] or b[1] > crop_box[3])
               for _, b in gt):
            continue
        crop = cv2.resize(img[y:y + size, x:x + size], (IMG_SIZE, IMG_SIZE))
        negatives.append((f"{lf.stem[:30]}_{x}_{y}", crop))
    return negatives
