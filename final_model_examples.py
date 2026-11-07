import random
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from pipelines import (CLASS_NAMES, IOU_MATCH, build_positives, get_yolo, iou, match,
                       source_of, yolo_detect)

# =========================================================
# Eksempelfigur: den endelige model på testbilleder (figur 17)
# ---------------------------------------------------------
# Kører YOLOv8n trænet ved 640 (batch 16 -- samme model som
# confusion matrix og per-klasse-tabellen i rapporten) på hele
# testdelen og vælger automatisk (seed 0) seks billeder, der
# viser forskellige billedtyper og klasser.
#
# Figur 1 (eksempler.png): billeder hvor ALLE skilte er fundet
#   med rigtig klasse og uden falske bokse.
# Figur 2 (fejl.png): billeder med fejl -- oversete skilte og
#   falske bokse -- til diskussionen af fx røde lys.
#
# Farver: grøn = rigtigt fund, rød = falsk/forkert boks,
#         gul stiplet = sandt skilt modellen overså.
# =========================================================

WEIGHTS = Path("results/yolov8n_640_b16_20261103_005628/weights/best.pt")
IMGSZ = 640
PANEL = 520  # visningsstørrelse pr. billede, så teksten kan læses i rapporten

GREEN, RED, YELLOW = (40, 180, 40), (40, 40, 230), (0, 210, 255)  # BGR

random.seed(0)
OUT_DIR = Path("results") / f"final_examples_{datetime.now():%Y%m%d_%H%M%S}"
(OUT_DIR / "single_images").mkdir(parents=True)

# ---------------------------------------------------------
# Kør modellen på alle testbilleder
# ---------------------------------------------------------
records = []
for name, img, gt in build_positives(10 ** 6):
    dets = yolo_detect(img, weights=WEIGHTS, imgsz=IMGSZ)
    matched, n_false = match(gt, dets)
    records.append(dict(name=name, img=img, gt=gt, dets=dets, matched=matched,
                        n_false=n_false, source=source_of(name),
                        perfect=n_false == 0 and all(m == g for m, (g, _) in zip(matched, gt))))
print(f"{len(records)} testbilleder, {sum(r['perfect'] for r in records)} helt uden fejl")


def classes(r):
    return {c for c, _ in r["gt"]}


def min_side(r):
    return min(min(b[2] - b[0], b[3] - b[1]) for _, b in r["gt"])


# ---------------------------------------------------------
# Vælg eksempler: én pr. "slot", forskellige billeder
# ---------------------------------------------------------
GOOD_SLOTS = [
    ("Fartskilt, vejfoto", lambda r: r["source"] == "Vejfoto (000xxx)" and classes(r) & set(range(2, 14))),
    ("Stopskilt", lambda r: 14 in classes(r)),
    ("Rødt lys", lambda r: 1 in classes(r) and r["source"] != "Simulator (fisheye)"),
    ("Grønt lys", lambda r: 0 in classes(r) and r["source"] != "Simulator (fisheye)"),
    ("Simulator, små objekter", lambda r: r["source"] == "Simulator (fisheye)" and min_side(r) < 16),
    ("Flere skilte i ét billede", lambda r: len(r["gt"]) >= 3),
]
BAD_SLOTS = [
    ("Overset rødt lys", lambda r: any(m is False and g == 1 for m, (g, _) in zip(r["matched"], r["gt"]))),
    ("Overset grønt lys", lambda r: any(m is False and g == 0 for m, (g, _) in zip(r["matched"], r["gt"]))),
    ("Falsk boks", lambda r: r["n_false"] > 0),
]


def pick(slots, pool):
    chosen, used = [], set()
    for label, cond in slots:
        for r in pool:
            if r["name"] not in used and cond(r):
                chosen.append((label, r))
                used.add(r["name"])
                break
    return chosen


good = pick(GOOD_SLOTS, [r for r in records if r["perfect"]])
bad = pick(BAD_SLOTS, [r for r in records if not r["perfect"]])


# ---------------------------------------------------------
# Tegn
# ---------------------------------------------------------
def draw(r):
    s = PANEL / max(r["img"].shape[:2])
    out = cv2.resize(r["img"], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

    # sande skilte der blev overset: gul stiplet
    for m, (g, b) in zip(r["matched"], r["gt"]):
        if m is False:
            x1, y1, x2, y2 = (int(v * s) for v in b)
            for i in range(x1, x2, 8):
                cv2.line(out, (i, y1), (min(i + 4, x2), y1), YELLOW, 2)
                cv2.line(out, (i, y2), (min(i + 4, x2), y2), YELLOW, 2)
            for i in range(y1, y2, 8):
                cv2.line(out, (x1, i), (x1, min(i + 4, y2)), YELLOW, 2)
                cv2.line(out, (x2, i), (x2, min(i + 4, y2)), YELLOW, 2)

    # detektioner: grøn hvis den rammer et sandt skilt med rigtig klasse, ellers rød
    W = out.shape[1]
    placed = []  # etiketter der allerede er tegnet, så de ikke lægges oven i hinanden
    labels = []
    for (box, cls, conf) in sorted(r["dets_conf"], key=lambda d: d[0][0]):
        hit = any(g == cls and iou(box, gb) >= IOU_MATCH for g, gb in r["gt"])
        color = GREEN if hit else RED
        x1, y1, x2, y2 = (int(v * s) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx = max(0, min(x1, W - tw - 4))  # hold etiketten inden for billedet
        # prøv over boksen, derefter under; ryk længere ned hvis den rammer en anden etiket
        candidates = [y1 - 4, y2 + th + 6] + [y2 + th + 6 + k * (th + 8) for k in range(1, 4)]
        for ty in candidates:
            rect = (lx, ty - th - 4, lx + tw + 4, ty + 3)
            if rect[1] >= 0 and not any(rect[0] < p[2] and p[0] < rect[2] and rect[1] < p[3] and p[1] < rect[3]
                                        for p in placed):
                break
        placed.append(rect)
        labels.append((rect, label, color, (lx, y1 if ty < y1 else y2)))
    # etiketterne tegnes til sidst, så ingen boks streger dem over
    for (rx1, ry1, rx2, ry2), label, color, anchor in labels:
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.putText(out, label, (rx1 + 2, ry2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# konfidens hentes separat (yolo_detect returnerer kun boks og klasse)
for _, r in good + bad:
    res = get_yolo(WEIGHTS).predict(r["img"], imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    r["dets_conf"] = [(tuple(b), int(c), float(p)) for b, c, p in
                      zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy())]


def save_grid(chosen, filename, cols):
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, r) in zip(axes.flat, chosen):
        im = draw(r)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=11)
        safe = "".join(ch if ch.isalnum() else "_" for ch in label)
        cv2.imwrite(str(OUT_DIR / "single_images" / f"{filename[:-4]}_{safe}.png"), im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


save_grid(good, "eksempler.png", cols=3)
save_grid(bad, "fejl.png", cols=3)

lines = [OUT_DIR.name, "", f"Model: {WEIGHTS} (imgsz {IMGSZ}, conf 0.25)",
         f"{len(records)} testbilleder, {sum(r['perfect'] for r in records)} helt uden fejl", "",
         "eksempler.png (alle skilte fundet med rigtig klasse, ingen falske bokse):"]
for label, r in good:
    found = ", ".join(f"{CLASS_NAMES[c]} {p:.2f}" for _, c, p in r["dets_conf"])
    lines.append(f"  {label}: {r['name']} [{r['source']}] -> {found}")
lines += ["", "fejl.png:"]
for label, r in bad:
    missed = [CLASS_NAMES[g] for m, (g, _) in zip(r["matched"], r["gt"]) if m is False]
    lines.append(f"  {label}: {r['name']} [{r['source']}] overset: {missed or '-'}, falske bokse: {r['n_false']}")
lines += ["", "Farver: grøn = rigtigt fund, rød = falsk/forkert boks, gul stiplet = overset sandt skilt."]
report = "\n".join(lines)
print(report)
(OUT_DIR / "summary.txt").write_text(report, encoding="utf-8")
print(f"\nGemt i {OUT_DIR}")
