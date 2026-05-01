import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

from astra_camera import AstraCamera

# Tkinter+PIL is used for display because cv2.imshow is broken on this
# Jetson Nano's opencv-python aarch64 wheel (window opens, image area is black).

model = YOLO("yolov8n.pt")
cam = AstraCamera()

FRAME_W = cam.width
FRAME_H = cam.height

cx1 = int(FRAME_W * 0.30)
cy1 = int(FRAME_H * 0.25)
cx2 = int(FRAME_W * 0.70)
cy2 = int(FRAME_H * 0.95)

CONF_THRESH = 0.45
DIST_STOP_M = 0.50
DIST_CAUTION_M = 1.20
DIST_DANGER_SCREEN_M = 0.35


def get_roi_depth_m(depth_mm, x1, y1, x2, y2):
    roi = depth_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        return float("inf")
    return float(np.median(valid)) / 1000.0


def box_intersects_center(x1, y1, x2, y2):
    return not (x2 < cx1 or x1 > cx2 or y2 < cy1 or y1 > cy2)


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    h, w = edges.shape
    polygon = np.array([[(0, h), (w, h), (w, h // 2), (0, h // 2)]])
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    return lines


def compute_steering(lines, frame_width):
    if lines is None:
        return 0
    x_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
    if not x_coords:
        return 0
    avg_x = np.mean(x_coords)
    steering = (avg_x - frame_width / 2) / (frame_width / 2)
    return np.clip(steering, -1, 1)


def draw_lanes(frame, lines):
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def decide_action(obstacle_status, steering_angle):
    if obstacle_status == "STOP":
        return {"throttle": 0.0, "steer": 0.0, "action": "STOP"}
    elif obstacle_status == "CAUTION":
        return {"throttle": 0.3, "steer": steering_angle, "action": "SLOW"}
    else:
        return {"throttle": 0.7, "steer": steering_angle, "action": "GO"}


def process_frame(color, depth_mm):
    frame = color
    lanes = detect_lanes(frame)
    steering_angle = compute_steering(lanes, FRAME_W)
    draw_lanes(frame, lanes)

    results = model(frame, verbose=False)[0]
    min_dist_m = float("inf")

    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id))
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dist_m = get_roi_depth_m(depth_mm, x1, y1, x2, y2)
            intersects = box_intersects_center(x1, y1, x2, y2)

            if intersects:
                min_dist_m = min(min_dist_m, dist_m)

            color_box = (0, 255, 0)
            if intersects and dist_m <= DIST_STOP_M:
                color_box = (0, 0, 255)
            elif intersects and dist_m <= DIST_CAUTION_M:
                color_box = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
            dist_str = f"{dist_m:.2f}m" if dist_m != float("inf") else "--"
            text = f"{label} {conf:.2f} D:{dist_str}"
            cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

    if min_dist_m <= DIST_STOP_M:
        status = "STOP"
        status_color = (0, 0, 255)
    elif min_dist_m <= DIST_CAUTION_M:
        status = "CAUTION"
        status_color = (0, 165, 255)
    else:
        status = "SAFE"
        status_color = (0, 255, 0)

    proximity = max(0.0, (DIST_CAUTION_M - min_dist_m) / DIST_CAUTION_M) if min_dist_m < float("inf") else 0.0
    action = decide_action(status, steering_angle)

    if min_dist_m <= DIST_DANGER_SCREEN_M:
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.putText(frame, "DANGER", (60, FRAME_H // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        dist_str = f"{min_dist_m:.2f}m" if min_dist_m != float("inf") else "--"
        cv2.putText(frame, f"DIST: {dist_str}", (90, FRAME_H // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    else:
        dist_str = f"{min_dist_m:.2f}m" if min_dist_m != float("inf") else "--"
        cv2.putText(frame, f"STATUS: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, f"DIST:   {dist_str}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"PROX:   {proximity:.2f}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"STEER:  {steering_angle:+.2f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"THROT:  {action['throttle']:.1f}", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"ACTION: {action['action']}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


def main():
    root = tk.Tk()
    root.title("YOLO Obstacle + Lane Detection")
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

        frame = process_frame(color, depth_mm)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
