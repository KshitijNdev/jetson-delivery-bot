import os
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from astra_camera import AstraCamera

# Save synchronized color + depth pairs so downstream models can train on
# either modality (or both). Within each category folder:
#   free_<ts>.jpg   — BGR color, JPEG
#   free_<ts>.png   — uint16 depth, PNG (lossless, raw millimetres preserved)
# Pairs share the same timestamp so they're trivial to associate.

FREE_DIR = os.path.expanduser("~/jetbot-project/data/collision/free")
BLOCKED_DIR = os.path.expanduser("~/jetbot-project/data/collision/blocked")

os.makedirs(FREE_DIR, exist_ok=True)
os.makedirs(BLOCKED_DIR, exist_ok=True)


def depth_to_color(depth_mm):
    depth_disp = np.clip(depth_mm, 600, 4000).astype(np.float32)
    depth_disp = ((depth_disp - 600) / (4000 - 600) * 255).astype(np.uint8)
    panel = cv2.applyColorMap(depth_disp, cv2.COLORMAP_TURBO)
    # Invalid pixels (depth==0) → black so "no data / too close" is visually
    # distinct from "very far".
    panel[depth_mm == 0] = 0
    return panel


def main():
    cam = AstraCamera()
    print("Press f for FREE, b for BLOCKED, q to quit")

    root = tk.Tk()
    root.title("Collect Collision Data — color + depth")

    label = tk.Label(root)
    label.pack()
    status = tk.Label(root, text="f=FREE  b=BLOCKED  q=QUIT", font=("DejaVu Sans", 12))
    status.pack()

    state = {
        "running": True,
        "img_ref": None,
        "last_color": None,
        "last_depth": None,
        "free_n": 0,
        "blocked_n": 0,
    }

    def on_quit(event=None):
        state["running"] = False
        root.quit()

    def save(category):
        color = state["last_color"]
        depth = state["last_depth"]
        if color is None or depth is None:
            return

        ts = int(time.time() * 1000)
        if category == "free":
            base = os.path.join(FREE_DIR, f"free_{ts}")
            state["free_n"] += 1
        else:
            base = os.path.join(BLOCKED_DIR, f"blocked_{ts}")
            state["blocked_n"] += 1

        cv2.imwrite(base + ".jpg", color)
        cv2.imwrite(base + ".png", depth)   # uint16 PNG preserves raw mm
        print(f"saved {base}.jpg + .png")

        status.configure(
            text=(
                f"saved {category.upper()}  |  free={state['free_n']} "
                f"blocked={state['blocked_n']}  |  f=FREE b=BLOCKED q=QUIT"
            )
        )

    root.bind("<Escape>", on_quit)
    root.bind("q", on_quit)
    root.bind("f", lambda e: save("free"))
    root.bind("b", lambda e: save("blocked"))
    root.protocol("WM_DELETE_WINDOW", on_quit)

    def update():
        if not state["running"]:
            return
        try:
            color, depth_mm = cam.read()
        except Exception as e:
            print("camera read error:", e)
            on_quit()
            return

        state["last_color"] = color
        state["last_depth"] = depth_mm

        depth_panel = depth_to_color(depth_mm)
        combined = np.hstack([color, depth_panel])
        rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo)
        state["img_ref"] = photo

        root.after(1, update)

    root.after(0, update)
    try:
        root.mainloop()
    finally:
        cam.release()


if __name__ == "__main__":
    main()
