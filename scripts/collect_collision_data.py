import os
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from astra_camera import AstraCamera

FREE_DIR = os.path.expanduser("~/jetbot-project/data/collision/free")
BLOCKED_DIR = os.path.expanduser("~/jetbot-project/data/collision/blocked")

os.makedirs(FREE_DIR, exist_ok=True)
os.makedirs(BLOCKED_DIR, exist_ok=True)


def main():
    cam = AstraCamera()
    print("Press f for FREE, b for BLOCKED, q to quit")

    root = tk.Tk()
    root.title("Collect Collision Data")
    label = tk.Label(root)
    label.pack()
    status = tk.Label(root, text="f=FREE  b=BLOCKED  q=QUIT", font=("DejaVu Sans", 12))
    status.pack()

    state = {"running": True, "img_ref": None, "last_color": None, "free_n": 0, "blocked_n": 0}

    def on_quit(event=None):
        state["running"] = False
        root.quit()

    def save(category):
        color = state["last_color"]
        if color is None:
            return
        if category == "free":
            fn = os.path.join(FREE_DIR, f"free_{int(time.time() * 1000)}.jpg")
            state["free_n"] += 1
        else:
            fn = os.path.join(BLOCKED_DIR, f"blocked_{int(time.time() * 1000)}.jpg")
            state["blocked_n"] += 1
        cv2.imwrite(fn, color)
        print("Saved", fn)
        status.configure(
            text=f"saved {category.upper()}  |  free={state['free_n']} blocked={state['blocked_n']}  |  f=FREE b=BLOCKED q=QUIT"
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
            color, _ = cam.read()
        except Exception as e:
            print("camera read error:", e)
            on_quit()
            return
        state["last_color"] = color

        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
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
