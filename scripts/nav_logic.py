"""Pure depth-zone navigation logic.

Imported by both the live YOLO+depth navigator (`yolo_obstacle_detection.py`)
and the simulator (`sim.py`) so they exercise identical decision code.
No camera or display dependencies — only numpy.

Convention: zones are ordered (LEFT, CENTER, RIGHT) from the bot's POV.
Distances are in metres. `float('inf')` means a zone has insufficient valid
depth data (which we treat as *blocked*, since on the Astra Pro it usually
indicates an obstacle closer than the 0.6 m min range).
"""
import numpy as np

# --- thresholds (metres) -----------------------------------------------------

DIST_STOP_M = 0.50
DIST_CAUTION_M = 1.50    # bumped from 1.2 → 1.5 so AVOID kicks in earlier and
                         # the bot has more headroom to complete a turn before
                         # the perpendicular wall closes in (late-turn fix).
DIST_DANGER_SCREEN_M = 0.35
AVOID_MIN_CLEARANCE_M = 0.70

# --- live-camera depth-frame analysis params --------------------------------

ZONE_MIN_VALID_PX = 200
BAND_TOP_FRAC = 0.30
BAND_BOT_FRAC = 0.85


def is_blocked(dist_m):
    """A zone is "blocked" if it reads close OR has insufficient valid depth."""
    return dist_m <= DIST_STOP_M or dist_m == float("inf")


def analyze_depth_zones(depth_mm):
    """For the live depth camera: split the central band of a depth_mm frame
    into LEFT/CENTER/RIGHT thirds, return (left_m, center_m, right_m) using
    the 5th-percentile of valid pixels in each zone."""
    H, W = depth_mm.shape
    y0 = int(H * BAND_TOP_FRAC)
    y1 = int(H * BAND_BOT_FRAC)
    band = depth_mm[y0:y1, :]
    zw = W // 3
    zones = []
    for i in range(3):
        x0 = i * zw
        x1 = (i + 1) * zw if i < 2 else W
        z = band[:, x0:x1]
        valid = z[z > 0]
        if valid.size < ZONE_MIN_VALID_PX:
            zones.append(float("inf"))
        else:
            zones.append(float(np.percentile(valid, 5)) / 1000.0)
    return tuple(zones)


def decide(zones):
    """Map (left_m, center_m, right_m) → action dict.

    Returns: {"status": "STOP"|"AVOID"|"GO", "throttle": float in [0,1],
              "steer": float in [-1,1] (negative = left), "action": str}
    """
    left_m, center_m, right_m = zones
    blocked = [is_blocked(d) for d in zones]
    n_blocked = sum(blocked)

    # 1. Everything blocked / unknown → STOP.
    if n_blocked == 3:
        return {"status": "STOP", "throttle": 0.0, "steer": 0.0, "action": "STOP"}

    # 2. Center blocked → must avoid toward a known-clear side.
    if blocked[1]:
        if not blocked[0] and not blocked[2]:
            steer = -0.6 if left_m > right_m else 0.6
        elif not blocked[0]:
            if left_m >= AVOID_MIN_CLEARANCE_M:
                steer = -0.6
            else:
                return {"status": "STOP", "throttle": 0.0, "steer": 0.0, "action": "STOP"}
        else:
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
        steer = 0.2
    elif blocked[2] and not blocked[0]:
        steer = -0.2
    elif not blocked[0] and not blocked[2]:
        steer = float(np.clip(
            (right_m - left_m) / max(left_m, right_m, 1.0),
            -0.3, 0.3,
        ))
    else:
        steer = 0.0
    return {"status": "GO", "throttle": 0.7, "steer": steer, "action": "GO"}


# --- recovery-aware navigator -----------------------------------------------

class Navigator:
    """Wraps `decide()` with a small state machine that handles "stuck"
    situations by reversing while turning. Use this anywhere you'd have
    called `decide()` directly when you also want recovery behavior.

    Stuck detection: STOP returned for `STUCK_FRAMES` consecutive ticks.
    Recovery: throttle=-0.4, full steer toward whichever side had more
    valid depth (or the larger reading if both are blocked), for
    `REVERSE_FRAMES` ticks. Then we re-evaluate.
    """

    STUCK_FRAMES = 8        # ~0.8 s of consecutive STOP → start reversing
    REVERSE_FRAMES = 18     # ~1.8 s of reverse-and-turn before re-evaluating
    REVERSE_THROTTLE = -0.4
    REVERSE_STEER = 1.0     # full turn while backing out

    def __init__(self):
        self._stop_streak = 0
        self._reverse_left = 0
        self._recovery_dir = 1.0  # +1 = turn right while reversing, -1 = left

    def step(self, zones):
        # 1. If we're already in a recovery cycle, keep reversing.
        if self._reverse_left > 0:
            self._reverse_left -= 1
            return {
                "status": "REVERSE",
                "throttle": self.REVERSE_THROTTLE,
                "steer": self._recovery_dir * self.REVERSE_STEER,
                "action": "REVERSE",
            }

        # 2. Otherwise run the stateless decision policy.
        d = decide(zones)

        # 3. Detect a sustained STOP and trigger recovery.
        if d["status"] == "STOP":
            self._stop_streak += 1
            if self._stop_streak >= self.STUCK_FRAMES:
                left_m, _, right_m = zones
                # Pick recovery direction: turn AWAY from whichever side has
                # less valid depth. If both are inf (totally blocked), default
                # to right; if both have values, turn toward the larger one.
                l_score = -1.0 if left_m == float("inf") else left_m
                r_score = -1.0 if right_m == float("inf") else right_m
                if l_score == r_score:
                    self._recovery_dir = 1.0  # bias right by default
                else:
                    self._recovery_dir = -1.0 if l_score > r_score else 1.0
                self._reverse_left = self.REVERSE_FRAMES
                self._stop_streak = 0
        else:
            self._stop_streak = 0

        return d

    def reset(self):
        self._stop_streak = 0
        self._reverse_left = 0
        self._recovery_dir = 1.0
