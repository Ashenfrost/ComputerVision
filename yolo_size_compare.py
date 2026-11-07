import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipelines import build_positives, match, yolo_detect

# =========================================================
# YOLOv8n trænet ved forskellige billedstørrelser
# ---------------------------------------------------------
# Opskalering skulle hjælpe på SMÅ skilte, fordi de fylder
# flere pixels i netværkets feature maps. Her måles det direkte:
# andel skilte fundet med rigtig klasse, opdelt efter størrelse,
# samt hvad det koster i hastighed.
#
# Kør:  python yolo_size_compare.py [--no-base] <kørselsmappe> [<kørselsmappe> ...]
# 416-kørslen (yolov8n) er med som udgangspunkt, medmindre --no-base gives.
# Model, imgsz og batch læses fra hver kørsels args.yaml, som Ultralytics
# gemmer under træning.
# =========================================================

RUN_416 = Path("results/yolov8n_20261102_163040")
RUN_ARGS = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
RUNS = RUN_ARGS if "--no-base" in sys.argv else [RUN_416] + RUN_ARGS


def run_args(run_dir):
    args = dict(line.split(": ", 1) for line in (run_dir / "args.yaml").read_text().splitlines() if ": " in line)
    model = Path(args["model"]).stem  # "yolov8n.pt" / "yolov8n-p2.yaml" -> "yolov8n" / "yolov8n-p2"
    return model, int(args["imgsz"]), int(args["batch"])


MODELS = {}
for run in RUNS:
    model, imgsz, batch = run_args(run)
    name = f"{model} {imgsz}" + ("" if batch == 32 else f" (batch {batch})")
    MODELS[name] = dict(weights=run / "weights" / "best.pt", imgsz=imgsz)
# faste farver i fast rækkefølge (validerede kategoriske farver)
STYLE = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
assert len(MODELS) <= len(STYLE), "højst 3 kørsler pr. plot -- del op i flere kørsler af scriptet"
BASE = next(iter(MODELS))

RUN_ID = f"yolo_size_compare_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
OUT_DIR.mkdir(parents=True)

SIZE_BINS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 10 ** 6)]
SIZE_NAMES = ["< 8", "8-16", "16-32", "32-64", "64-128", ">= 128"]

random.seed(0)
all_test = build_positives(10 ** 6)  # alle testbilleder med skilte (samme som robustness_test.py)

by_size = {m: defaultdict(lambda: [0, 0]) for m in MODELS}
for name, img, gt in all_test:
    for m, cfg in MODELS.items():
        matched, _ = match(gt, yolo_detect(img, **cfg))
        for (gc, gb), dc in zip(gt, matched):
            side = min(gb[2] - gb[0], gb[3] - gb[1])
            b = next(i for i, (lo, hi) in enumerate(SIZE_BINS) if lo <= side < hi)
            by_size[m][b][0] += dc is not False and dc == gc
            by_size[m][b][1] += 1

W = max(len(m) for m in MODELS) + 2
lines = [RUN_ID, ""] + [f"{m}: {cfg['weights']}" for m, cfg in MODELS.items()] + ["",
         "== Andel skilte fundet med rigtig klasse, efter størrelse (korteste side i px) ==",
         f"{'':<10}" + "".join(f"{m:>{W}}" for m in MODELS) + f"{'n':>7}"]
for i, g in enumerate(SIZE_NAMES):
    row = f"{g:<10}"
    for m in MODELS:
        h, t = by_size[m][i]
        row += f"{100 * h / t:>{W - 1}.1f}%"
    lines.append(row + f"{by_size[BASE][i][1]:>7}")
tot = {m: sum(v[0] for v in by_size[m].values()) / sum(v[1] for v in by_size[m].values()) for m in MODELS}
lines.append(f"{'I alt':<10}" + "".join(f"{100 * tot[m]:>{W - 1}.1f}%" for m in MODELS))
lines.append("")

# hastighed, samme metode som i robustness_test.py
speed_imgs = [img for _, img, _ in all_test[:100]]
lines.append("== Hastighed (100 billeder, gennemsnit pr. billede) ==")
for m, cfg in MODELS.items():
    for device, label in ((0, "GPU"), ("cpu", "CPU")):
        for im in speed_imgs[:10]:  # opvarmning
            yolo_detect(im, device=device, **cfg)
        t0 = time.perf_counter()
        for im in speed_imgs:
            yolo_detect(im, device=device, **cfg)
        ms = 1000 * (time.perf_counter() - t0) / len(speed_imgs)
        lines.append(f"{m + ' (' + label + ')':<{W + 8}}{ms:>8.1f} ms {1000 / ms:>6.0f} billeder/s")

report = "\n".join(lines)
print(report)
(OUT_DIR / "summary.txt").write_text(report, encoding="utf-8")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0"})
fig, ax = plt.subplots(figsize=(8 + len(MODELS), 4.2))
x = np.arange(len(SIZE_NAMES))
width = 0.8 / len(MODELS)
for j, m in enumerate(MODELS):
    vals = [100 * by_size[m][i][0] / by_size[m][i][1] for i in range(len(SIZE_NAMES))]
    bars = ax.bar(x + (j - (len(MODELS) - 1) / 2) * width, vals, width * 0.92, label=m, color=STYLE[m])
    ax.bar_label(bars, fmt="%.0f", fontsize=8, color="#52514e", padding=2)
n = [by_size[BASE][i][1] for i in range(len(SIZE_NAMES))]
ax.set_xticks(x, [f"{g}\n(n={k})" for g, k in zip(SIZE_NAMES, n)])
ax.set_xlabel("Skiltets korteste side (pixels)")
ax.set_ylabel("Fundet med rigtig klasse (%)")
ax.set_ylim(0, 105)
ax.grid(axis="x", visible=False)
ax.set_title(" vs. ".join(m.split(" (")[0] for m in MODELS), loc="left")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(MODELS))
fig.tight_layout()
fig.savefig(OUT_DIR / "stoerrelse.png", dpi=150)
print(f"\nGemt i {OUT_DIR}")
