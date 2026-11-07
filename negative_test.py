import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipelines import (CLASS_NAMES, IMG_SIZE, IOU_MATCH, RED_CLASSES, build_positives,
                       build_road_negatives, build_strawberries, classical_detect,
                       cnn_classify, match, yolo_detect)

# =========================================================
# Test på POSITIVE og NEGATIVE billeder
# ---------------------------------------------------------
# Negative billeder = billeder UDEN skilte. De måler falske
# alarmer: hvor tit siger metoden "skilt!" når der ikke er et?
# Positive billeder måler om metoden overhovedet finder skilte
# (ellers kunne en metode der aldrig finder noget få 0 fejl).
#
# Metoder (defineret i pipelines.py):
#   1. Klassisk A/B/C:  median blur -> rød HSV -> morfologi -> Hough circles (-> ring-tjek)
#   2. Klassisk C + CNN: cirklerne fra C skæres ud og klassificeres af CNN'et
#   3. YOLOv8n trænet ved 416 og ved 640 (den endelige model)
# =========================================================

YOLO_640_WEIGHTS = Path("results/yolov8n_640_20261102_184940/weights/best.pt")

random.seed(0)
np.random.seed(0)

N_POSITIVE = 150         # tilfældige testbilleder med skilte
N_ROAD_NEGATIVE = 100    # vejudsnit uden skilte
CNN_CONF = 0.90          # ekstra forsøg: afvis CNN-svar under 90% sikkerhed

RUN_ID = f"negative_test_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
OUT_DIR.mkdir(parents=True)

# rækkefølgen betyder noget: samme seed + samme rækkefølge -> samme billeder
positives = build_positives(N_POSITIVE)
strawberries = build_strawberries()
road_negatives = build_road_negatives(N_ROAD_NEGATIVE)
print(f"Positive: {len(positives)}, jordbær: {len(strawberries)}, vejudsnit: {len(road_negatives)}")


# ---------------------------------------------------------
# Kør alle metoder
# ---------------------------------------------------------
METHODS = ["Klassisk A (naiv)", "Klassisk B (strammere Hough)", "Klassisk C (+ ring-tjek)",
           "Klassisk C + CNN", f"Klassisk C + CNN (conf>={CNN_CONF})", "YOLOv8n 416", "YOLOv8n 640"]
SHOWN = ("Klassisk A (naiv)", "Klassisk C + CNN", "YOLOv8n 640")  # kolonner i eksempelbillederne
NO_CLASS = {"Klassisk A (naiv)", "Klassisk B (strammere Hough)", "Klassisk C (+ ring-tjek)"}


def run_all(img):
    """Returnerer {metode: [(boks, klasse eller None)]}."""
    circles_c = classical_detect(img, "C")
    labelled = [(b, *cnn_classify(img, b)) for b in circles_c]
    return {
        "Klassisk A (naiv)": [(b, None) for b in classical_detect(img, "A")],
        "Klassisk B (strammere Hough)": [(b, None) for b in classical_detect(img, "B")],
        "Klassisk C (+ ring-tjek)": [(b, None) for b in circles_c],
        "Klassisk C + CNN": [(b, c) for b, c, _ in labelled if c is not None],
        f"Klassisk C + CNN (conf>={CNN_CONF})": [(b, c) for b, c, p in labelled if c is not None and p >= CNN_CONF],
        "YOLOv8n 416": yolo_detect(img),
        "YOLOv8n 640": yolo_detect(img, weights=YOLO_640_WEIGHTS, imgsz=640),
    }


def draw(img, dets, color):
    out = img.copy()
    for b, c in dets:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if c is not None:
            cv2.putText(out, CLASS_NAMES[c], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


lines = [RUN_ID, ""]

# --- negative ---
for set_name, images in (("Jordbær", strawberries), ("Vejudsnit", road_negatives)):
    fp_imgs = {m: 0 for m in METHODS}
    fp_total = {m: 0 for m in METHODS}
    examples = []
    for name, img in images:
        res = run_all(img)
        for m in METHODS:
            fp_total[m] += len(res[m])
            fp_imgs[m] += len(res[m]) > 0
        if len(examples) < 6 and (res["Klassisk A (naiv)"] or res["YOLOv8n 640"]):
            row = [draw(img, res[m], (0, 0, 255)) for m in SHOWN]
            row = [cv2.resize(r, (IMG_SIZE, IMG_SIZE)) for r in row]
            examples.append(np.hstack(row))
    lines.append(f"== NEGATIVE: {set_name} ({len(images)} billeder) ==")
    lines.append(f"{'Metode':<36} {'billeder m. falsk alarm':>24} {'falske alarmer i alt':>21}")
    for m in METHODS:
        lines.append(f"{m:<36} {fp_imgs[m]:>4} / {len(images):<4} ({100*fp_imgs[m]/len(images):5.1f}%)   {fp_total[m]:>10}")
    lines.append("")
    if examples:
        cv2.imwrite(str(OUT_DIR / f"eksempler_{set_name.lower()}.jpg"), np.vstack(examples))

# --- positive ---
found = {m: 0 for m in METHODS}
correct_cls = {m: 0 for m in METHODS}
false_pos = {m: 0 for m in METHODS}
found_red = {m: 0 for m in METHODS}
n_gt = n_gt_red = 0
pos_examples = []
for name, img, gt in positives:
    res = run_all(img)
    n_gt += len(gt)
    n_gt_red += sum(c in RED_CLASSES for c, _ in gt)
    for m in METHODS:
        matched, fp = match(gt, res[m])
        for (gc, _), dc in zip(gt, matched):
            if dc is not False:
                found[m] += 1
                found_red[m] += gc in RED_CLASSES
                correct_cls[m] += dc == gc
        false_pos[m] += fp
    if len(pos_examples) < 6:
        row = [draw(img, res[m], (0, 0, 255)) for m in SHOWN]
        pos_examples.append(np.hstack(row))
cv2.imwrite(str(OUT_DIR / "eksempler_positive.jpg"), np.vstack(pos_examples))

lines.append(f"== POSITIVE: {len(positives)} testbilleder, {n_gt} skilte ({n_gt_red} med rød kant) ==")
lines.append(f"{'Metode':<36} {'fundet (alle)':>14} {'fundet (røde)':>14} {'rigtig klasse':>14} {'falske bokse':>13}")
for m in METHODS:
    cls_txt = "-" if m in NO_CLASS else f"{correct_cls[m]:>4} ({100*correct_cls[m]/n_gt:4.1f}%)"
    lines.append(f"{m:<36} {found[m]:>4} ({100*found[m]/n_gt:4.1f}%) "
                 f"{found_red[m]:>4} ({100*found_red[m]/n_gt_red:4.1f}%) {cls_txt:>14} {false_pos[m]:>13}")
lines.append("")
lines.append(f"'fundet' = en detektion overlapper skiltet med IoU >= {IOU_MATCH}.")
lines.append("'rigtig klasse' = fundet OG klassen er korrekt. Klassisk A/B/C alene giver ingen klasse.")
lines.append("'falske bokse' på positive billeder = detektioner der ikke rammer noget skilt.")
lines.append("Eksempelbilleder: kolonner = Klassisk A | Klassisk C + CNN | YOLOv8n 640.")

report = "\n".join(lines)
print(report)
(OUT_DIR / "summary.txt").write_text(report, encoding="utf-8")
print(f"\nGemt i {OUT_DIR}")
