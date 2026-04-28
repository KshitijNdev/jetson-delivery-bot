import cv2
import os
import time
from astra_camera import AstraCamera

FREE_DIR = os.path.expanduser("~/jetbot-project/data/collision/free")
BLOCKED_DIR = os.path.expanduser("~/jetbot-project/data/collision/blocked")

os.makedirs(FREE_DIR, exist_ok=True)
os.makedirs(BLOCKED_DIR, exist_ok=True)

cam = AstraCamera()
print("Press f for FREE, b for BLOCKED, q to quit")

try:
    while True:
        color, _ = cam.read()

        display = color.copy()
        cv2.putText(display, "f=FREE  b=BLOCKED  q=QUIT", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Collision Data", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("f"):
            filename = os.path.join(FREE_DIR, f"free_{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, color)
            print("Saved", filename)
        elif key == ord("b"):
            filename = os.path.join(BLOCKED_DIR, f"blocked_{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, color)
            print("Saved", filename)

finally:
    cam.release()
    cv2.destroyAllWindows()
