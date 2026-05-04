import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from astra_camera import AstraCamera

# Depth-first navigation, YOLO secondary for object labelling.
# Decisions (STOP / AVOID / GO + steering) are computed from the depth frame
# zoned LEFT / CENTER / RIGHT. YOLO runs on the color frame to identify
# *what* objects are visible (e.g. for delivery-target recognition), but
# does not influence steering or throttle.

model = YOLO("yolov8n.pt")
cam = AstraCamera()

FRAME_W = cam.width
FRAME_H = cam.height

# Vertical band of the depth frame used for navigation decisions. We ignore
# the top (ceiling/sky) and bottom (immediate floor) so distant obstacles at
# robot height drive the decision.
BAND_TOP_FRAC = 0.30
BAND_BOT_FRAC = 0.85

# Distance thresholds (metres).
DIST_STOP_M = 0.50
DIST_CAUTION_M = 1.20
DIST_DANGER_SCREEN_M = 0.35

# Zone sampling: 5th-percentile so a few noisy near-zero pixels don't
# trigger false stops. Need at least this many valid pixels per zone for
# the reading to count.
ZONE_MIN_VALID_PX = 200

CONF_THRESH = 0.45


def analyze_depth_zones(depth_mm):
    """Returns (left_m, center_m, right_m) — nearest obstacle per zone in metres.
    `inf` means insufficient valid data (treat as safe)."""
    y0 = int(FRAME_H * BAND_TOP_FRAC)
    y1 = int(FRAME_H * BAND_BOT_FRAC)
    band = depth_mm[y0:y1, :]

    zw = FRAME_W // 3
    zones = []
    for i in range(3):
        x0 = i * zw
        x1 = (i + 1) * zw if i < 2 else FRAME_W
        z = band[:, x0:x1]
        valid = z[z > 0]
        if valid.size < ZONE_MIN_VALID_PX:
            zones.append(float("inf"))
        else:
            zones.append(float(np.percentile(valid, 5)) / 1000.0)
    return tuple(zones)


# Minimum clearance a side must show before we'll commit to steering into it
# when the center is blocked. Below this we'd rather STOP than turn into a
# space we're not sure is wide enough.
AVOID_MIN_CLEARANCE_M = 0.70


def is_blocked(dist_m):
    """A zone is "blocked" if it reads close OR has insufficient valid depth.
    `inf` (no valid pixels) almost always means an obstacle is closer than
    the Astra Pro's 0.6m minimum range, which makes the IR pattern saturate
    and the depth read as zero — i.e. *more* dangerous, not safer."""
    return dist_m <= DIST_STOP_M or dist_m == float("inf")


def decide(zones):
    left_m, center_m, right_m = zones
    blocked = [is_blocked(d) for d in zones]
    n_blocked = sum(blocked)

    # 1. Everything blocked / unknown → STOP.
    if n_blocked == 3:
        return {"status": "STOP", "throttle": 0.0, "steer": 0.0, "action": "STOP"}

    # 2. Center blocked → must avoid.
    if blocked[1]:
        if not blocked[0] and not blocked[2]:
            # Both sides open: turn toward the wider one.
            steer = -0.6 if left_m > right_m else 0.6
        elif not blocked[0]:
            # Only LEFT is known clear — commit only if it has real room.
            if left_m >= AVOID_MIN_CLEARANCE_M:
                steer = -0.6
            else:
                return {"status": "STOP", "throttle": 0.0, "steer": 0.0, "action": "STOP"}
        else:
            # Only RIGHT is known clear.
            if right_m >= AVOID_MIN_CLEARANCE_M:
                steer = 0.6
            else:
                return {"status": "STOP", "throttle": 0.0, "steer": 0.0, "action": "STOP"}
        return {"status": "AVOID", "throttle": 0.3, "steer": steer, "action": "AVOID"}

    # 3. Center open but tight → slow down and steer toward more space.
    if center_m <= DIST_CAUTION_M:
        if not blocked[0] and not blocked[2]:
            steer = -0.6 if left_m > right_m else 0.6
        elif not blocked[0]:
            steer = -0.6
        elif not blocked[2]:
            steer = 0.6
        else:
            steer = 0.0
        return {"status": "AVOID", "throttle": 0.3, "steer": steer, "action": "AVOID"}

    # 4. Center clear. Bias gently toward whichever side has more headroom,
    # and bias *away* from a side we have no data on (treat unknown as risky).
    if blocked[0] and not blocked[2]:
        steer = 0.2   # left is unknown/close → favor right
    elif blocked[2] and not blocked[0]:
        steer = -0.2  # right is unknown/close → favor left
    elif not blocked[0] and not blocked[2]:
        steer = float(np.clip(
            (right_m - left_m) / max(left_m, right_m, 1.0),
            -0.3, 0.3,
        ))
    else:
        steer = 0.0  # both sides unknown but center clear — go straight, slow nothing
    return {"status": "GO", "throttle": 0.7, "steer": steer, "action": "GO"}


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
        "STOP":  (0, 0, 255),
        "AVOID": (0, 165, 255),
        "GO":    (0, 255, 0),
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
        decision = decide(zones)

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
