import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from astra_camera import AstraCamera

# cv2.imshow is broken on the Jetson Nano's bundled opencv-python aarch64 wheel
# (window opens but image area renders black). Tkinter + PIL gives us a working
# display path that doesn't depend on the cv2 Qt plugin at all.


def main():
    cam = AstraCamera()

    root = tk.Tk()
    root.title("Astra Pro - RGB | Depth")
    label = tk.Label(root)
    label.pack()

    state = {"frames": 0, "start": time.time(), "running": True, "save": False, "img_ref": None}

    def on_quit(event=None):
        state["running"] = False
        root.quit()

    def on_save(event=None):
        state["save"] = True

    root.bind("<Escape>", on_quit)
    root.bind("q", on_quit)
    root.bind("s", on_save)
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

        state["frames"] += 1
        elapsed = time.time() - state["start"]
        fps = state["frames"] / elapsed if elapsed > 0 else 0

        depth_disp = np.clip(depth_mm, 600, 4000).astype(np.float32)
        depth_disp = ((depth_disp - 600) / (4000 - 600) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_disp, cv2.COLORMAP_TURBO)

        cy, cx = cam.height // 2, cam.width // 2
        center_d = int(depth_mm[cy, cx])
        center_str = f"{center_d / 1000:.2f}m" if center_d > 0 else "--"
        cv2.circle(depth_color, (cx, cy), 6, (255, 255, 255), 2)
        cv2.putText(depth_color, center_str, (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if state["save"]:
            state["save"] = False
            ts = int(time.time())
            cv2.imwrite(f"frame_color_{ts}.jpg", color)
            cv2.imwrite(f"frame_depth_{ts}.png", depth_mm)
            print(f"Saved frame_color_{ts}.jpg and frame_depth_{ts}.png")

        combined = np.hstack([color, depth_color])
        rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=photo)
        state["img_ref"] = photo  # keep ref alive so Tk doesn't GC it

        root.after(1, update)

    root.after(0, update)
    try:
        root.mainloop()
    finally:
        cam.release()


if __name__ == "__main__":
    main()
