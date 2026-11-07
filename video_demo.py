import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from pipelines import CLASS_NAMES, YOLO_CONF, classical_detect, cnn_classify, get_yolo

# =========================================================
# Video-demo til fremlæggelsen
# ---------------------------------------------------------
# Kører den endelige model (YOLOv8n trænet ved 640) på en video
# og viser bokse, klassenavne og en FPS-tæller, så man kan SE
# at det kører i realtid. Med --classical vises den klassiske
# pipeline (trin C + CNN) ved siden af til sammenligning.
#
# Kør:
#   python video_demo.py                      YOLO 640 på testvideo.mp4
#   python video_demo.py --classical          YOLO og klassisk side om side
#   python video_demo.py --save               gem også en video (backup til fremlæggelsen)
#   python video_demo.py --cpu                kør YOLO på CPU (viser at det IKKE er realtid)
#   python video_demo.py --model=416          brug 416-modellen i stedet
#   python video_demo.py --fast               afspil så hurtigt som muligt (ikke låst til videoens fps)
#   python video_demo.py --no-show --save     lav kun den gemte video, uden vindue
#   python video_demo.py andenvideo.mp4       en anden video
#
# Taster i vinduet: mellemrum = pause, q eller Esc = stop.
#
# Vinduet vises med tkinter i stedet for cv2.imshow, fordi den
# OpenCV-version Ultralytics installerer (opencv-python-headless)
# ikke kan åbne vinduer.
# =========================================================

args = [a for a in sys.argv[1:] if not a.startswith("--")]
VIDEO = Path(args[0]) if args else Path("testvideo.mp4")
CLASSICAL = "--classical" in sys.argv
SAVE = "--save" in sys.argv
SHOW = "--no-show" not in sys.argv
FAST = "--fast" in sys.argv
DEVICE = "cpu" if "--cpu" in sys.argv else 0
MODEL_SIZE = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--model=")), 640)

WEIGHTS = {
    416: Path("results/yolov8n_20261102_163040/weights/best.pt"),
    640: Path("results/yolov8n_640_20261102_184940/weights/best.pt"),
}[MODEL_SIZE]

DISPLAY_HEIGHT = 640  # videoen er lille (416 px), så den skaleres op på skærmen
YOLO_COLOR = (0, 200, 0)        # grøn (BGR)
CLASSICAL_COLOR = (255, 160, 0)  # blå


def yolo_frame(model, frame):
    r = model.predict(frame, imgsz=MODEL_SIZE, conf=YOLO_CONF, verbose=False, device=DEVICE)[0]
    return [(b, int(c), float(p)) for b, c, p in
            zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def classical_frame(frame):
    dets = []
    for box in classical_detect(frame, "C"):
        cls, conf = cnn_classify(frame, box)
        if cls is not None:
            dets.append((box, cls, conf))
    return dets


def draw(frame, dets, color, title, ms_history):
    """Tegner bokse på et opskaleret billede, så teksten kan læses på en projektor."""
    scale = DISPLAY_HEIGHT / frame.shape[0]
    out = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    for (x1, y1, x2, y2), cls, conf in dets:
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 8), (p1[0] + tw + 6, p1[1]), color, -1)
        cv2.putText(out, label, (p1[0] + 3, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # FPS = hvor mange billeder metoden KAN nå pr. sekund (gennemsnit af de sidste 30)
    ms = sum(ms_history) / len(ms_history)
    header = [title, f"{ms:5.1f} ms/billede  =  {1000 / ms:5.1f} FPS"]
    cv2.rectangle(out, (0, 0), (out.shape[1], 62), (0, 0, 0), -1)
    for i, text in enumerate(header):
        cv2.putText(out, text, (10, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Kan ikke åbne videoen: {VIDEO}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{VIDEO}: {int(cap.get(3))}x{int(cap.get(4))}, {video_fps:.0f} fps, {n_frames} billeder")

    model = get_yolo(WEIGHTS)
    device_name = "CPU" if DEVICE == "cpu" else "GPU"
    yolo_title = f"YOLOv8n {MODEL_SIZE} ({device_name})"

    # opvarmning: første kald er langsomt (CUDA starter, modellen indlæses)
    ok, frame = cap.read()
    for _ in range(5):
        yolo_frame(model, frame)
        if CLASSICAL:
            classical_frame(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = out_dir = None
    if SAVE:
        out_dir = Path("results") / f"video_demo_{VIDEO.stem}_{MODEL_SIZE}_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True)

    yolo_ms, classical_ms = deque(maxlen=30), deque(maxlen=30)
    all_yolo_ms, all_classical_ms = [], []
    writer_box = [None]

    def next_canvas():
        """Læser næste billede, kører modellerne og returnerer det tegnede billede (None ved slut)."""
        ok, frame = cap.read()
        if not ok:
            return None
        t0 = time.perf_counter()
        dets = yolo_frame(model, frame)
        yolo_ms.append(1000 * (time.perf_counter() - t0))
        all_yolo_ms.append(yolo_ms[-1])
        panels = [draw(frame, dets, YOLO_COLOR, yolo_title, yolo_ms)]

        if CLASSICAL:
            t0 = time.perf_counter()
            dets_c = classical_frame(frame)
            classical_ms.append(1000 * (time.perf_counter() - t0))
            all_classical_ms.append(classical_ms[-1])
            panels.insert(0, draw(frame, dets_c, CLASSICAL_COLOR, "Klassisk C + CNN", classical_ms))

        canvas = np.hstack(panels)
        if SAVE:
            if writer_box[0] is None:
                h, w = canvas.shape[:2]
                writer_box[0] = cv2.VideoWriter(str(out_dir / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                video_fps, (w, h))
            writer_box[0].write(canvas)
        return canvas

    if SHOW:
        root = tk.Tk()
        root.title("Trafikskilte - video demo")
        root.configure(bg="black")
        label = tk.Label(root, bg="black")
        label.pack()
        state = {"paused": False}

        def tick():
            if state["paused"]:
                root.after(50, tick)
                return
            t_frame = time.perf_counter()
            canvas = next_canvas()
            if canvas is None:
                root.destroy()
                return
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
            label.configure(image=photo)
            label.image = photo  # behold en reference, ellers rydder Python billedet væk
            # Vent så videoen afspilles i normal hastighed (medmindre --fast eller modellen er for langsom)
            spent = time.perf_counter() - t_frame
            wait = 1 if FAST else max(1, int(1000 * (1 / video_fps - spent)))
            root.after(wait, tick)

        def on_key(event):
            if event.keysym in ("q", "Escape"):
                root.destroy()
            elif event.keysym == "space":
                state["paused"] = not state["paused"]

        root.bind("<Key>", on_key)
        root.after(0, tick)
        root.mainloop()
    else:
        while next_canvas() is not None:
            pass

    cap.release()
    if writer_box[0] is not None:
        writer_box[0].release()

    lines = [f"{VIDEO}: {len(all_yolo_ms)} billeder, video {video_fps:.0f} fps",
             f"{yolo_title}: {np.mean(all_yolo_ms):.1f} ms/billede = {1000 / np.mean(all_yolo_ms):.1f} FPS"]
    if CLASSICAL:
        lines.append(f"Klassisk C + CNN: {np.mean(all_classical_ms):.1f} ms/billede "
                     f"= {1000 / np.mean(all_classical_ms):.1f} FPS")
    realtime = 1000 / np.mean(all_yolo_ms) >= video_fps
    lines.append(f"YOLO {'KAN' if realtime else 'kan IKKE'} følge med videoens {video_fps:.0f} fps.")
    report = "\n".join(lines)
    print(report)
    if out_dir is not None:
        (out_dir / "summary.txt").write_text(report, encoding="utf-8")
        print(f"Gemt i {out_dir}")


if __name__ == "__main__":
    main()
