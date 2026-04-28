import cv2
import numpy as np
import time
from astra_camera import AstraCamera

cam = AstraCamera()
frames = 0
start = time.time()

try:
    while True:
        color, depth_mm = cam.read()

        frames += 1
        elapsed = time.time() - start
        fps = frames / elapsed if elapsed > 0 else 0

        # Depth colormap: normalize valid range (600-4000 mm) to 0-255
        depth_display = np.clip(depth_mm, 600, 4000).astype(np.float32)
        depth_display = ((depth_display - 600) / (4000 - 600) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_TURBO)

        # Center-pixel depth readout
        cy, cx = cam.height // 2, cam.width // 2
        center_d = depth_mm[cy, cx]
        center_str = f"{center_d / 1000:.2f}m" if center_d > 0 else "--"
        cv2.circle(depth_color, (cx, cy), 6, (255, 255, 255), 2)
        cv2.putText(depth_color, center_str, (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(color, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack([color, depth_color])
        cv2.imshow("Astra Pro — RGB | Depth", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            cv2.imwrite(f"frame_color_{ts}.jpg", color)
            cv2.imwrite(f"frame_depth_{ts}.png", depth_mm)
            print(f"Saved frame_color_{ts}.jpg and frame_depth_{ts}.png")

finally:
    cam.release()
    cv2.destroyAllWindows()
