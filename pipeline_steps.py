import random
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from pipelines import (CLASS_NAMES, RED_CLASSES, build_positives, build_strawberries,
                       classical_detect, cnn_classify, match, source_of)

# =========================================================
# Mellemresultater i den klassiske pipeline (trin C)
# ---------------------------------------------------------
# Viser billedet efter HVERT filter, så man kan se hvad hvert
# trin bidrager med -- og hvor det går galt:
#   1. stort skilt (vejfoto)     -> virker
#   2. lille skilt (vejfoto)     -> forsvinder i masken / Hough
#   3. jordbær, afvist           -> ring-tjekket redder os
#   4. jordbær, falsk alarm      -> ring-tjekket er ikke nok
# Eksemplerne vælges automatisk (seed 0), så figuren kan genskabes.
# =========================================================

RUN_ID = f"pipeline_steps_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
(OUT_DIR / "single_images").mkdir(parents=True)

GT_COLOR = (0, 220, 255)     # gul (BGR): det sande skilt
HOUGH_COLOR = (255, 160, 0)  # blå: alle Hough-cirkler
KEPT_COLOR = (0, 200, 0)     # grøn: beholdt efter ring-tjek
REJECT_COLOR = HOUGH_COLOR   # tynd blå: afvist af ring-tjek (rød kan ikke ses på jordbær)


def run(img):
    steps = {}
    boxes = classical_detect(img, stage="C", steps=steps)
    return steps, boxes


def min_side(box):
    return min(box[2] - box[0], box[3] - box[1])


# ---------------------------------------------------------
# Vælg eksempler
# ---------------------------------------------------------
random.seed(0)
positives = build_positives(10 ** 6)
big = small = None
for name, img, gt in positives:
    if source_of(name) != "Vejfoto (000xxx)":
        continue
    red = [(c, b) for c, b in gt if c in RED_CLASSES]
    if not red:
        continue
    found, _ = match(red, [(b, None) for b in run(img)[1]])
    if big is None and min_side(red[0][1]) >= 40 and found[0] is not False:
        big = (name, img, red)
    if small is None and min_side(red[0][1]) < 24 and found[0] is False:
        small = (name, img, red)
    if big and small:
        break

rejected = false_alarm = None
for name, img in build_strawberries():
    steps, boxes = run(img)
    if rejected is None and steps["circles"] and not boxes:
        rejected = (name, img, [])
    if false_alarm is None and boxes:
        false_alarm = (name, img, [])
    if rejected and false_alarm:
        break

examples = [
    ("Stort skilt", big),
    ("Lille skilt", small),
    ("Jordbær, afvist", rejected),
    ("Jordbær, falsk alarm", false_alarm),
]
examples = [(label, ex) for label, ex in examples if ex is not None]


# ---------------------------------------------------------
# Byg billederne for hvert trin
# ---------------------------------------------------------
def panels_for(img, gt):
    steps, boxes = run(img)

    original = img.copy()
    for _, (x1, y1, x2, y2) in gt:
        cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), GT_COLOR, 2)

    hough = img.copy()
    for x, y, r in steps["circles"]:
        cv2.circle(hough, (int(x), int(y)), int(r), HOUGH_COLOR, 2)

    final = img.copy()
    kept = {(float(x), float(y), float(r)) for x, y, r in steps["kept"]}
    labels = []
    for x, y, r in steps["circles"]:
        if (float(x), float(y), float(r)) in kept:
            continue
        cv2.circle(final, (int(x), int(y)), int(r), REJECT_COLOR, 1)
    for box, (x, y, r) in zip(boxes, steps["kept"]):
        cv2.circle(final, (int(x), int(y)), int(r), KEPT_COLOR, 2)
        cls, conf = cnn_classify(img, box)
        labels.append(f"{CLASS_NAMES[cls]} ({conf:.2f})" if cls is not None else "?")

    n_all, n_kept = len(steps["circles"]), len(steps["kept"])
    return [
        ("1. Original", original, None),
        ("2. Median blur (5)", steps["median"], None),
        ("3. Rød HSV-maske", steps["hsv_mask"], "gray"),
        ("4. Close (5x5)", steps["close"], "gray"),
        ("5. Open (5x5)", steps["open"], "gray"),
        ("6. Gaussian blur (5)", steps["gaussian"], "gray"),
        (f"7. Hough: {n_all} cirkler", hough, None),
        (f"8. Ring-tjek: {n_kept} tilbage", final, None),
    ], labels


rows = [(label, ex[0], *panels_for(ex[1], ex[2])) for label, ex in examples]


def show(ax, title, im, cmap):
    if cmap is None:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_single(folder, title, im, cmap):
    safe = "".join(c if c.isalnum() else "_" for c in title).strip("_")
    cv2.imwrite(str(folder / f"{safe}.png"), im)


# samlet figur: én række pr. eksempel
n_cols = len(rows[0][2])
fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.2 * n_cols, 2.5 * len(rows)), squeeze=False)
lines = [RUN_ID, "", "Klassisk pipeline, trin C (samme parametre som negative_test.py)", ""]
for i, (label, name, panels, cnn_labels) in enumerate(rows):
    for j, (title, im, cmap) in enumerate(panels):
        show(axes[i][j], title, im, cmap)
    axes[i][0].text(-0.08, 0.5, label, transform=axes[i][0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=10)
    if cnn_labels:
        axes[i][-1].set_xlabel("CNN: " + ", ".join(cnn_labels), fontsize=8)
        axes[i][-1].axis("on")
        axes[i][-1].set_xticks([])
        axes[i][-1].set_yticks([])

    # én figur pr. eksempel + hvert trin som enkeltbillede
    folder = OUT_DIR / "single_images" / f"{i + 1}_{label.replace(' ', '_').replace(',', '')}"
    folder.mkdir()
    for title, im, cmap in panels:
        save_single(folder, title, im, cmap)
    f2, a2 = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.8))
    for ax, (title, im, cmap) in zip(a2, panels):
        show(ax, title, im, cmap)
    f2.suptitle(f"{label} ({name})", fontsize=10, x=0.01, ha="left")
    f2.tight_layout()
    f2.savefig(OUT_DIR / f"eksempel_{i + 1}.png", dpi=150, bbox_inches="tight")
    plt.close(f2)

    lines.append(f"{i + 1}. {label}: {name}")
    lines.append(f"   Hough-cirkler: {panels[6][0].split(': ')[1]}, efter ring-tjek: {panels[7][0].split(': ')[1]}")
    if cnn_labels:
        lines.append(f"   CNN: {', '.join(cnn_labels)}")

lines += ["", "Farver: gul = sandt skilt, blå = Hough-cirkel, grøn = beholdt, tynd blå (trin 8) = afvist af ring-tjek."]
fig.tight_layout()
fig.savefig(OUT_DIR / "pipeline_steps.png", dpi=150, bbox_inches="tight")

report = "\n".join(lines)
print(report)
(OUT_DIR / "summary.txt").write_text(report, encoding="utf-8")
print(f"\nGemt i {OUT_DIR}")
