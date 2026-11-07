import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipelines import (build_positives, build_road_negatives, build_strawberries,
                       classical_detect, cnn_classify, match, source_of, yolo_detect)

# =========================================================
# Robusthed, hastighed og parametersweep
# ---------------------------------------------------------
#   1a. Lys:        hvordan klarer metoderne sig i mørkere/lysere billeder?
#   1b. Størrelse:  hvor små skilte kan hver metode finde?
#   1c. Kilde:      forskel på GTSRB-udsnit, vejfotos, simulator osv.
#   4.  Hastighed:  ms pr. billede på GPU og CPU (kan det køre i realtid i en bil?)
#   5.  Hough-sweep: effekten af param2 med og uden ring-tjek -- på VALIDERINGS-
#       sættet, så parametrene ikke vælges på de data de bagefter testes på.
# =========================================================

RUN_ID = f"robustness_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
OUT_DIR.mkdir(parents=True)

METHODS = ["Klassisk C", "Klassisk C + CNN", "YOLOv8n"]
# validerede kategoriske farver (blå, orange, aqua) + forskellige markører,
# så linjerne også kan skelnes uden farvesyn og på sort/hvid print
STYLE = {
    "Klassisk C": dict(color="#2a78d6", marker="o"),
    "Klassisk C + CNN": dict(color="#eb6834", marker="s"),
    "YOLOv8n": dict(color="#1baf7a", marker="^"),
}
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 8,
})

lines = [RUN_ID, ""]


def log(text=""):
    print(text)
    lines.append(text)


def run3(img):
    """De tre metoder. Klassisk C giver kun bokse; CNN og YOLO giver også klasse."""
    circles = classical_detect(img, "C")
    return {
        "Klassisk C": [(b, None) for b in circles],
        "Klassisk C + CNN": [(b, c) for b in circles for c, _ in [cnn_classify(img, b)] if c is not None],
        "YOLOv8n": yolo_detect(img),
    }


def hits(gt, dets, method):
    """Pr. sandt skilt: True hvis metoden ramte det. For CNN og YOLO skal klassen også være rigtig."""
    matched, _ = match(gt, dets)
    if method == "Klassisk C":
        return [m is not False for m in matched]
    return [m is not False and m == gc for (gc, _), m in zip(gt, matched)]


Y_LABEL = "Andel skilte ramt (%)"
Y_NOTE = "Klassisk C: fundet. Klassisk C + CNN og YOLOv8n: fundet med rigtig klasse."


# ---------------------------------------------------------
# 1a. Lys
# ---------------------------------------------------------
random.seed(0)
positives = build_positives(150)  # samme 150 billeder som i negative_test.py
FACTORS = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]

light = {m: [] for m in METHODS}
for f in FACTORS:
    counts = defaultdict(int)
    n = 0
    for _, img, gt in positives:
        # gang alle pixels med f: < 1 = mørkere (skumring), > 1 = lysere (overeksponering)
        adj = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        res = run3(adj)
        n += len(gt)
        for m in METHODS:
            counts[m] += sum(hits(gt, res[m], m))
    for m in METHODS:
        light[m].append(100 * counts[m] / n)

log("== 1a. LYS (150 testbilleder, 185 skilte) ==")
log(f"{'Lysfaktor':<20}" + "".join(f"{f:>8}" for f in FACTORS))
for m in METHODS:
    log(f"{m:<20}" + "".join(f"{v:>7.1f}%" for v in light[m]))
log(Y_NOTE)
log()

fig, ax = plt.subplots(figsize=(7, 4.2))
for m in METHODS:
    ax.plot(FACTORS, light[m], label=m, **STYLE[m])
ax.axvline(1.0, color="#8a8984", linestyle="--", linewidth=1)
ax.text(1.02, 3, "original", color="#52514e", fontsize=9)
ax.set_xlabel("Lysfaktor (pixelværdier ganget med)")
ax.set_ylabel(Y_LABEL)
ax.set_ylim(0, 100)
ax.set_title("Robusthed over for lys", loc="left")
ax.legend(frameon=False)
fig.text(0.01, 0.01, Y_NOTE, fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT_DIR / "lys.png", dpi=150)

# eksempel: ét billede ved hver lysfaktor
_, ex_img, _ = positives[2]
strip = [np.clip(ex_img.astype(np.float32) * f, 0, 255).astype(np.uint8) for f in FACTORS]
for s, f in zip(strip, FACTORS):
    cv2.putText(s, f"x{f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.imwrite(str(OUT_DIR / "lys_eksempel.jpg"), np.hstack([cv2.resize(s, (208, 208)) for s in strip]))


# ---------------------------------------------------------
# 1b + 1c. Størrelse og kilde -- på ALLE testbilleder med skilte
# ---------------------------------------------------------
random.seed(0)
all_test = build_positives(10 ** 6)
SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 10 ** 6)]
SIZE_NAMES = ["< 8", "8-16", "16-32", "32-64", "64-128", ">= 128"]
by_size = {m: defaultdict(lambda: [0, 0]) for m in METHODS}   # bin -> [ramt, i alt]
by_source = {m: defaultdict(lambda: [0, 0]) for m in METHODS}

for name, img, gt in all_test:
    res = run3(img)
    src = source_of(name)
    for m in METHODS:
        for (gc, gb), h in zip(gt, hits(gt, res[m], m)):
            side = min(gb[2] - gb[0], gb[3] - gb[1])  # korteste side i pixels
            b = next(i for i, (lo, hi) in enumerate(SIZE_BINS) if lo <= side < hi)
            by_size[m][b][0] += h
            by_size[m][b][1] += 1
            by_source[m][src][0] += h
            by_source[m][src][1] += 1


def grouped_bars(groups, data, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(groups))
    width = 0.26
    for i, m in enumerate(METHODS):
        vals = [100 * data[m][g][0] / data[m][g][1] if data[m][g][1] else 0 for g in range(len(groups))]
        ax.bar(x + (i - 1) * width, vals, width - 0.03, label=m, color=STYLE[m]["color"])
    n = [data[METHODS[0]][g][1] for g in range(len(groups))]
    ax.set_xticks(x, [f"{g}\n(n={k})" for g, k in zip(groups, n)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.01, 0.01, Y_NOTE, fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / filename, dpi=150)


def table(title, groups, data, keys):
    log(title)
    log(f"{'':<22}" + "".join(f"{m:>20}" for m in METHODS) + f"{'n':>7}")
    for g, k in zip(groups, keys):
        row = f"{g:<22}"
        for m in METHODS:
            h, t = data[m][k]
            row += f"{(100 * h / t if t else 0):>19.1f}%"
        log(row + f"{data[METHODS[0]][k][1]:>7}")
    log(Y_NOTE)
    log()


n_imgs = len(all_test)
n_signs = sum(len(gt) for _, _, gt in all_test)
table(f"== 1b. STØRRELSE (korteste side i px; {n_imgs} testbilleder, {n_signs} skilte) ==",
      SIZE_NAMES, by_size, range(len(SIZE_BINS)))
grouped_bars(SIZE_NAMES, by_size, "Fundne skilte efter størrelse", "Skiltets korteste side (pixels)", "stoerrelse.png")

sources = sorted(by_source["YOLOv8n"])
src_idx = {m: {i: by_source[m][s] for i, s in enumerate(sources)} for m in METHODS}
table("== 1c. KILDE ==", sources, src_idx, range(len(sources)))
grouped_bars(sources, src_idx, "Fundne skilte efter datakilde", "Kilde", "kilde.png")


# ---------------------------------------------------------
# 4. Hastighed
# ---------------------------------------------------------
speed_imgs = [img for _, img, _ in positives[:100]]
SPEED_METHODS = [
    ("Klassisk C (CPU)", lambda im: classical_detect(im, "C")),
    ("Klassisk C + CNN (GPU)", lambda im: [cnn_classify(im, b, "cuda") for b in classical_detect(im, "C")]),
    ("Klassisk C + CNN (CPU)", lambda im: [cnn_classify(im, b, "cpu") for b in classical_detect(im, "C")]),
    ("YOLOv8n (GPU)", lambda im: yolo_detect(im, device=0)),
    ("YOLOv8n (CPU)", lambda im: yolo_detect(im, device="cpu")),
]
log("== 4. HASTIGHED (100 billeder, 416x416, gennemsnit pr. billede) ==")
log(f"{'Metode':<26}{'ms/billede':>12}{'billeder/s':>12}")
for name, fn in SPEED_METHODS:
    for im in speed_imgs[:10]:  # opvarmning: første kald er langsomme (indlæsning, CUDA-opstart)
        fn(im)
    t0 = time.perf_counter()
    for im in speed_imgs:
        fn(im)
    ms = 1000 * (time.perf_counter() - t0) / len(speed_imgs)
    log(f"{name:<26}{ms:>12.1f}{1000 / ms:>12.0f}")
log("Klassisk C bruger kun OpenCV på CPU. Tider inkluderer forbehandling, ikke indlæsning af billedet.")
log()


# ---------------------------------------------------------
# 5. Hough-sweep på VALIDERINGSsættet
# ---------------------------------------------------------
random.seed(1)
val_pos = build_positives(150, split="valid")
val_neg = build_road_negatives(100, split="valid") + build_strawberries()
PARAM2 = [10, 15, 20, 25, 30, 35, 40, 50]
CHOSEN = 30  # værdien brugt i trin B og C

sweep = {}
for ring in (False, True):
    rec, fpi = [], []
    for p2 in PARAM2:
        found = total = 0
        for _, img, gt in val_pos:
            matched, _ = match(gt, [(b, None) for b in classical_detect(img, "C", param2=p2, ring_check=ring)])
            found += sum(m is not False for m in matched)
            total += len(gt)
        fp = sum(len(classical_detect(img, "C", param2=p2, ring_check=ring)) for _, img in val_neg)
        rec.append(100 * found / total)
        fpi.append(fp / len(val_neg))
    sweep[ring] = (rec, fpi)

n_val_signs = sum(len(gt) for _, _, gt in val_pos)
log(f"== 5. HOUGH-SWEEP (valideringssæt: {len(val_pos)} positive / {n_val_signs} skilte, "
    f"{len(val_neg)} negative inkl. 40 jordbær; minDist=40) ==")
log(f"{'param2':<26}" + "".join(f"{p:>7}" for p in PARAM2))
for ring in (False, True):
    rec, fpi = sweep[ring]
    tag = "med ring-tjek" if ring else "uden ring-tjek"
    log(f"{'fundet % ' + tag:<26}" + "".join(f"{v:>7.1f}" for v in rec))
    log(f"{'falske/negativt ' + tag:<26}" + "".join(f"{v:>7.2f}" for v in fpi))
log()

SWEEP_STYLE = {False: dict(color="#2a78d6", marker="o", label="Uden ring-tjek (trin B)"),
               True: dict(color="#eb6834", marker="s", label="Med ring-tjek (trin C)")}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
for ring in (False, True):
    ax1.plot(PARAM2, sweep[ring][0], **SWEEP_STYLE[ring])
    ax2.plot(PARAM2, sweep[ring][1], **SWEEP_STYLE[ring])
for ax in (ax1, ax2):
    ax.axvline(CHOSEN, color="#8a8984", linestyle="--", linewidth=1)
    ax.set_xlabel("Hough param2 (højere = kræver mere bevis for en cirkel)")
ax1.set_ylabel("Skilte fundet (%)")
ax1.set_ylim(0, 100)
ax1.set_title("Positive billeder", loc="left")
ax2.set_ylabel("Falske alarmer pr. negativt billede")
ax2.set_yscale("log")
ax2.set_title("Negative billeder (log-skala)", loc="left")
ax1.legend(frameon=False)
fig.text(0.01, 0.01, f"Valideringssæt. Stiplet linje = valgt værdi ({CHOSEN}).", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT_DIR / "hough_sweep.png", dpi=150)

(OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nGemt i {OUT_DIR}")
