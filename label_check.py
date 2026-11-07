import os
import re
import random
from pathlib import Path
from collections import Counter

import yaml
import cv2

# =========================================================
# Tjek af label-kvalitet i cardata/
# ---------------------------------------------------------
# Spørgsmål: når en label-fil indeholder flere forskellige
# klasser, er det så ægte flere skilte i billedet, eller er
# det samme skilt labelet to gange (dobbelt-label)?
# Metode: IoU (Intersection over Union) mellem bokse.
#   IoU ~ 1  -> boksene dækker det samme område = samme skilt
#   IoU ~ 0  -> boksene ligger forskellige steder = forskellige skilte
# =========================================================

DATASET_ROOT = Path("cardata")
SPLITS = ["train", "valid", "test"]
IOU_DUPLICATE = 0.5      # over denne grænse regnes to bokse som "samme skilt"
NUM_EXAMPLES = 12        # antal eksempelbilleder der gemmes til visuel kontrol
OUT_DIR = Path("label_check")

with open(DATASET_ROOT / "data.yaml") as f:
    CLASS_NAMES = yaml.safe_load(f)["names"]


def read_boxes(label_path):
    """Returnerer liste af (klasse, x1, y1, x2, y2) i normaliserede koordinater."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou(a, b):
    """Overlap-areal divideret med samlet areal af to bokse."""
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[3] - a[1]) * (a[4] - a[2])
    area_b = (b[3] - b[1]) * (b[4] - b[2])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def source_of(filename):
    """Datasættet er samlet fra flere kilder -- kan ses på filnavnet."""
    if re.match(r"\d{5}_\d{5}_", filename):
        return "GTSRB-udsnit"        # opskalerede, slørede nærbilleder af ét skilt
    if filename.startswith("FisheyeCamera"):
        return "Simulator (fisheye)"  # syntetiske billeder fra bilsimulator
    if filename.startswith("road"):
        return "road*.png"
    return "Vejfoto (000xxx)"


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


def draw_example(image_path, boxes, out_path):
    """Tegner alle bokse med klassenavn, så man selv kan vurdere labelen."""
    img = cv2.imread(str(image_path))
    H, W = img.shape[:2]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for i, (c, x1, y1, x2, y2) in enumerate(boxes):
        color = colors[i % len(colors)]
        p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, CLASS_NAMES[c], (p1[0], max(12, p1[1] - 4 - 14 * i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(out_path), img)


OUT_DIR.mkdir(exist_ok=True)
random.seed(0)

for split in SPLITS:
    labels_dir = DATASET_ROOT / split / "labels"
    images_dir = DATASET_ROOT / split / "images"

    n_files = n_empty = n_multi_box = n_multi_class = 0
    n_conflict_files = 0         # forskellige klasser på (næsten) samme sted
    n_same_class_dup_files = 0   # samme klasse to gange på samme sted
    n_real_multi_files = 0       # flere klasser, men forskellige steder
    gtsrb_style = conflict_gtsrb = 0
    files_per_source = Counter()
    boxes_per_source = Counter()
    tiny_per_source = Counter()  # bokse under 8 px (fjernes i Machineconvoluted_crops.py)
    conflict_pairs = Counter()   # hvilke klasser bliver blandet sammen
    conflict_examples = []
    real_multi_examples = []

    for label_file in sorted(os.listdir(labels_dir)):
        if not label_file.endswith(".txt"):
            continue
        n_files += 1
        boxes = read_boxes(labels_dir / label_file)
        source = source_of(label_file)
        is_gtsrb = source == "GTSRB-udsnit"
        gtsrb_style += is_gtsrb
        files_per_source[source] += 1
        for b in boxes:
            boxes_per_source[source] += 1
            # alle billeder er 416x416, så normaliseret * 416 = pixels
            if min(b[3] - b[1], b[4] - b[2]) * 416 < 8:
                tiny_per_source[source] += 1
        if not boxes:
            n_empty += 1
            continue
        if len(boxes) > 1:
            n_multi_box += 1
        classes = {b[0] for b in boxes}
        if len(classes) > 1:
            n_multi_class += 1

        has_conflict = has_same_dup = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) < IOU_DUPLICATE:
                    continue
                if boxes[i][0] != boxes[j][0]:
                    has_conflict = True
                    pair = tuple(sorted((CLASS_NAMES[boxes[i][0]], CLASS_NAMES[boxes[j][0]])))
                    conflict_pairs[pair] += 1
                else:
                    has_same_dup = True

        stem = label_file[:-4]
        if has_conflict:
            n_conflict_files += 1
            conflict_gtsrb += is_gtsrb
            conflict_examples.append(stem)
        elif len(classes) > 1:
            n_real_multi_files += 1
            real_multi_examples.append(stem)
        if has_same_dup:
            n_same_class_dup_files += 1

    print(f"\n===== {split} =====")
    print(f"Label-filer i alt:                         {n_files}")
    print("  Fordeling på kilde (filer / bokse / bokse under 8 px):")
    for src, n in files_per_source.most_common():
        print(f"    {src:<22} {n:>5} / {boxes_per_source[src]:>5} / {tiny_per_source[src]:>4}")
    print(f"  tomme (ingen skilte = negative billeder): {n_empty}")
    print(f"  med mere end én boks:                    {n_multi_box}")
    print(f"  med mere end én KLASSE:                  {n_multi_class}")
    print(f"    -> konflikt (forsk. klasse, IoU>={IOU_DUPLICATE}): {n_conflict_files}"
          f"  (heraf GTSRB-agtige: {conflict_gtsrb})")
    print(f"    -> ægte flere skilte (lavt overlap):   {n_real_multi_files}")
    print(f"  samme klasse dobbelt på samme sted:      {n_same_class_dup_files}")
    if conflict_pairs:
        print("  Hyppigste konflikt-par:")
        for (a, b), n in conflict_pairs.most_common(10):
            print(f"    {a:<16} <-> {b:<16} {n}")

    # gem tilfældige eksempler af begge typer til visuel kontrol
    for kind, stems in (("konflikt", conflict_examples), ("aegte_multi", real_multi_examples)):
        for stem in random.sample(stems, min(NUM_EXAMPLES // 2, len(stems))):
            img_path = find_image(images_dir, stem)
            if img_path:
                out = OUT_DIR / f"{split}_{kind}_{stem[:40]}.jpg"
                draw_example(img_path, read_boxes(labels_dir / (stem + ".txt")), out)

print(f"\nEksempelbilleder gemt i {OUT_DIR}/")
