import numpy as np
import cv2
from primesense import openni2

WIDTH = 640
HEIGHT = 480
FPS = 30

# Directory containing libOpenNI2.so on Jetson (ARM64 Debian layout)
_OPENNI2_LIB_PATH = "/usr/lib/aarch64-linux-gnu"


class AstraCamera:
    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS):
        self.width = width
        self.height = height
        openni2.initialize(_OPENNI2_LIB_PATH)
        self._dev = openni2.Device.open_any()

        if self._dev.is_image_registration_mode_supported(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
        ):
            self._dev.set_image_registration_mode(
                openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
            )

        self._color = self._dev.create_color_stream()
        self._color.set_video_mode(openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
            resolutionX=width, resolutionY=height, fps=fps,
        ))
        self._color.start()

        self._depth = self._dev.create_depth_stream()
        self._depth.set_video_mode(openni2.VideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
            resolutionX=width, resolutionY=height, fps=fps,
        ))
        self._depth.start()

    def read(self):
        """Returns (color_bgr uint8 HxWx3, depth_mm uint16 HxW). depth 0 = invalid."""
        cf = self._color.read_frame()
        df = self._depth.read_frame()
        color = np.frombuffer(cf.get_buffer_as_uint8(), dtype=np.uint8)
        color_bgr = cv2.cvtColor(
            color.reshape(self.height, self.width, 3).copy(),
            cv2.COLOR_RGB2BGR,
        )
        depth_mm = np.frombuffer(df.get_buffer_as_uint16(), dtype=np.uint16)
        depth_mm = depth_mm.reshape(self.height, self.width).copy()
        return color_bgr, depth_mm

    def release(self):
        self._color.stop()
        self._depth.stop()
        self._dev.close()
        openni2.unload()
