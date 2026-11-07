import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO, settings

# =========================================================
# YOLOv8: detektion OG klassifikation i ét netværk
# ---------------------------------------------------------
# Modsat CNN-iterationerne skal YOLO selv finde skiltet i
# billedet (bounding box) og samtidig sige hvilken klasse det er.
# Vi starter fra yolov8n.pt, som er fortrænet på COCO-datasættet
# (der bl.a. indeholder "stop sign" og "traffic light"), og
# finjusterer den på cardata -- det kaldes transfer learning.
# =========================================================

# Kør med "--smoke" for en hurtig test (1 epoch, 10% af data),
# der bare tjekker at alt virker, før den lange træning startes.
SMOKE_TEST = "--smoke" in sys.argv

# Billedstørrelse under træning og test. Billederne er 416x416, men
# kør med fx "--imgsz=640" for at opskalere: små skilte fylder så flere
# pixels i netværkets feature maps, men hver epoch tager ca. (640/416)^2
# = 2,4 gange så lang tid.
IMGSZ = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--imgsz=")), 416)

# Batch-størrelse. 32 som standard; ved store imgsz kan GPU'ens 6 GB
# løbe tør, så kør fx "--batch=16". Afviger den fra 32, kommer den
# med i mappenavnet, så man kan se at forsøget ikke er helt sammenligneligt.
BATCH = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--batch=")), 32)

# Arkitektur. Kør fx "--model=yolov8s" eller "--model=yolov8n-p2":
#   yolov8n     standard, mindste model (udgangspunkt)
#   yolov8s     samme opbygning, ca. 3x så mange parametre -> mere KAPACITET
#   yolov8n-p2  yolov8n med et ekstra detektionslag ved stride 4 (felter på
#               4x4 px i stedet for mindst 8x8) -> finere OPLØSNING til små objekter
# P2 findes ikke fortrænet, så den bygges fra sin .yaml, og alle lag der
# matcher yolov8n.pt (rygraden m.m.) får COCO-vægtene. Det nye lag starter fra bunden.
MODEL = next((a.split("=")[1] for a in sys.argv if a.startswith("--model=")), "yolov8n")

DATA_YAML = Path("cardata/data_yolo.yaml")  # rettede stier, se kommentar i filen

# kørsler før imgsz-argumentet hedder bare "yolov8n_<tid>" og er 416
RUN_NAME = f"yolo_smoketest_{MODEL}_{IMGSZ}" if SMOKE_TEST else f"{MODEL}_{IMGSZ}"
if BATCH != 32:
    RUN_NAME += f"_b{BATCH}"
RUN_ID = f"{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
RESULTS_DIR = Path("results").resolve()

# Ultralytics logger kun til TensorBoard hvis det er slået til.
# Event-filen havner i results/<RUN_ID>/ sammen med resten.
settings.update({"tensorboard": True})


def main():
    if MODEL.endswith("-p2"):
        model = YOLO(f"{MODEL}.yaml").load("yolov8n.pt")
    else:
        model = YOLO(f"{MODEL}.pt")  # downloades automatisk første gang

    model.train(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        epochs=1 if SMOKE_TEST else 50,
        fraction=0.1 if SMOKE_TEST else 1.0,
        patience=15,          # stop tidligt hvis val ikke forbedres i 15 epochs
        batch=BATCH,
        workers=4,
        seed=0,               # samme resultat hver gang -> sammenlignelige forsøg
        # samme augmentation-fravalg som i Machineconvoluted_crops.py:
        fliplr=0.0,           # ingen spejlvending -- tal bliver forkerte
        hsv_h=0.0,            # ingen hue-ændring -- rød/grøn er selve forskellen
        project=str(RESULTS_DIR),
        name=RUN_ID,
        exist_ok=False,       # overskriv aldrig en eksisterende mappe
    )

    # Evaluering på TEST-sættet (som modellen aldrig har set under træning).
    # Valideringssættet er brugt til at vælge den bedste epoch, så test
    # giver et mere ærligt tal.
    best = YOLO(RESULTS_DIR / RUN_ID / "weights" / "best.pt")
    metrics = best.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,          # test i samme størrelse som der er trænet i
        project=str(RESULTS_DIR),
        name=f"{RUN_ID}_test",
        exist_ok=False,
    )

    # per-klasse tal til rapporten
    lines = [f"{RUN_ID} -- test-sæt", f"mAP50:    {metrics.box.map50:.3f}",
             f"mAP50-95: {metrics.box.map:.3f}", "",
             f"{'Klasse':<16} {'P':>6} {'R':>6} {'mAP50':>6}"]
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, _ = metrics.box.class_result(i)
        lines.append(f"{best.names[int(c)]:<16} {p:>6.3f} {r:>6.3f} {ap50:>6.3f}")
    report = "\n".join(lines)
    print(report)
    with open(RESULTS_DIR / f"{RUN_ID}_test" / "per_class_test.txt", "w", encoding="utf-8") as f:
        f.write(report)


# På Windows kræver workers > 0 denne guard (ellers starter
# hver worker-proces hele træningen forfra)
if __name__ == "__main__":
    main()
