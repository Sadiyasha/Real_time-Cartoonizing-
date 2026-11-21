"""
Live webcam cartoonizer.

Modes:
  1 = AnimeGANv2 (pretrained neural network)
  2 = Fast OpenCV cartoon filter
  3 = Original camera

Keys:
  1 / 2 / 3 = switch mode
  s         = save a snapshot
  q         = quit
"""
import time

import cv2
import numpy as np
import torch

# Lower = faster, higher = better quality. Try 384 or 512 if you have a GPU.
SIZE = 256
CAMERA_ID = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading AnimeGANv2 (the first run downloads the weights)...")
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
).to(device).eval()


@torch.no_grad()
def anime(frame):
    """Run one BGR frame through AnimeGANv2 and return a BGR frame."""
    h, w = frame.shape[:2]
    new_h = SIZE
    new_w = max(8, int(SIZE * w / h) // 8 * 8)  # keep aspect ratio, multiple of 8
    small = cv2.resize(frame, (new_w, new_h))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = (
        torch.from_numpy(rgb).float().div(127.5).sub(1)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    y = model(x)[0].clamp(-1, 1).add(1).mul(127.5).byte()
    y = y.permute(1, 2, 0).cpu().numpy()
    return cv2.resize(cv2.cvtColor(y, cv2.COLOR_RGB2BGR), (w, h))


def fast_cartoon(frame, levels=8):
    """Quick OpenCV cartoon: smooth, reduce colors, add edges."""
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    smooth = cv2.bilateralFilter(small, 9, 75, 75)
    step = 256 // levels
    quant = ((smooth // step) * step + step // 2).astype(np.uint8)
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7
    )
    out = cv2.bitwise_and(quant, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    return cv2.resize(out, (frame.shape[1], frame.shape[0]))


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: could not open camera. Try CAMERA_ID = 1.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mode = 1
    names = {1: "AnimeGANv2", 2: "OpenCV cartoon", 3: "Original"}
    fps, last = 0.0, time.time()

    print("Running. Keys: 1/2/3 = mode, s = save, q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: cannot read frame")
            break
        frame = cv2.flip(frame, 1)  # mirror view

        if mode == 1:
            out = anime(frame)
        elif mode == 2:
            out = fast_cartoon(frame)
        else:
            out = frame

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        cv2.putText(
            out, f"{names[mode]} | {fps:.1f} FPS", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow("Cartoon Camera", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            mode = int(chr(key))
        elif key == ord("s"):
            name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(name, out)
            print(f"Saved {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()