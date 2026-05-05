import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from astra_camera import AstraCamera
from nav_logic import (
    Navigator, analyze_depth_zones,
    DIST_STOP_M, DIST_CAUTION_M, DIST_DANGER_SCREEN_M,
    BAND_TOP_FRAC, BAND_BOT_FRAC,
)

# Depth-first navigation, YOLO secondary for object labelling.
# Decision logic lives in `nav_logic` so the simulator and the live navigator
# exercise the same code. YOLO runs on the color frame to identify *what*
# objects are visible (e.g. for delivery-target recognition), but does not
# influence steering or throttle.

model = YOLO("yolov8n.pt")
cam = AstraCamera()
nav = Navigator()

FRAME_W = cam.width
FRAME_H = cam.height

CONF_THRESH = 0.45


def get_roi_depth_m(depth_mm, x1, y1, x2, y2):
    roi = depth_mm[max(0, y1):y2, max(0, x1):x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return float("inf")
    return float(np.median(valid)) / 1000.0


def annotate_color(frame, results, depth_mm):
    """Draw YOLO bounding boxes + labels on the color frame.
    Labelling-only — no danger coloring, since depth owns the decisions."""
    if results.boxes is None:
        return
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESH:
            continue
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        dist_m = get_roi_depth_m(depth_mm, x1, y1, x2, y2)
        dist_str = f"{dist_m:.2f}m" if dist_m != float("inf") else "--"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)
        cv2.putText(frame, f"{label} {dist_str}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def render_depth_panel(depth_mm, zones, decision):
    """Color depth frame with zone overlay and distance readouts."""
    depth_disp = np.clip(depth_mm, 600, 4000).astype(np.float32)
    depth_disp = ((depth_disp - 600) / (4000 - 600) * 255).astype(np.uint8)
    panel = cv2.applyColorMap(depth_disp, cv2.COLORMAP_TURBO)
    # Invalid pixels (depth==0) → black, so "no data" is visually distinct
    # from "very close" (which would otherwise both render dark purple).
    panel[depth_mm == 0] = 0

    y0 = int(FRAME_H * BAND_TOP_FRAC)
    y1 = int(FRAME_H * BAND_BOT_FRAC)
    cv2.rectangle(panel, (0, y0), (FRAME_W - 1, y1), (255, 255, 255), 1)

    zw = FRAME_W // 3
    zone_names = ("L", "C", "R")
    for i, (name, dist) in enumerate(zip(zone_names, zones)):
        x0 = i * zw
        x1_ = (i + 1) * zw if i < 2 else FRAME_W
        cv2.line(panel, (x0, y0), (x0, y1), (255, 255, 255), 1)
        dist_str = f"{dist:.2f}m" if dist != float("inf") else "--"
        cv2.putText(panel, f"{name}: {dist_str}", (x0 + 6, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Status banner
    status_color = {
        "STOP":    (0, 0, 255),
        "AVOID":   (0, 165, 255),
        "GO":      (0, 255, 0),
        "REVERSE": (255, 100, 200),
    }.get(decision["status"], (255, 255, 255))
    cv2.putText(panel, decision["status"], (10, FRAME_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(panel, f"thr={decision['throttle']:.2f} steer={decision['steer']:+.2f}",
                (110, FRAME_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def render_full_screen_danger(combined, center_m):
    overlay = combined.copy()
    overlay[:] = (0, 0, 255)
    out = cv2.addWeighted(overlay, 0.65, combined, 0.35, 0)
    h, w = out.shape[:2]
    cv2.putText(out, "DANGER", (w // 2 - 180, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
    dist_str = f"{center_m:.2f}m" if center_m != float("inf") else "--"
    cv2.putText(out, f"DIST: {dist_str}", (w // 2 - 130, h // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    return out


def main():
    root = tk.Tk()
    root.title("YOLO + Depth Navigation")
    label = tk.Label(root)
    label.pack()
    state = {"running": True, "img_ref": None}

    def on_quit(event=None):
        state["running"] = False
        root.quit()

    root.bind("<Escape>", on_quit)
    root.bind("q", on_quit)
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

        zones = analyze_depth_zones(depth_mm)
        decision = nav.step(zones)

        results = model(color, verbose=False)[0]
        annotate_color(color, results, depth_mm)

        depth_panel = render_depth_panel(depth_mm, zones, decision)
        combined = np.hstack([color, depth_panel])

        if zones[1] <= DIST_DANGER_SCREEN_M:
            combined = render_full_screen_danger(combined, zones[1])

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
