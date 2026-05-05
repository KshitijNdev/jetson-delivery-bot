"""Top-down simulator for the depth-zone navigation logic.

Loads the Light Engineering 2nd-floor map, places a virtual JetBot, and
ray-casts a synthetic depth view from its pose at every step. The synthetic
L/C/R zone distances are fed straight into `nav_logic.decide()` — the SAME
function the live `yolo_obstacle_detection.py` calls — so this is a real
test of the navigation policy, not a parallel implementation.

Controls:
    Space   pause / resume
    R       reset to starting pose
    Click   teleport bot to that point on the map
    Q/Esc   quit
"""
import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from building_map import build_grid, RESOLUTION_M, WALL
from nav_logic import Navigator, DIST_STOP_M, DIST_CAUTION_M

# --- physics ---------------------------------------------------------------

V_MAX = 0.40            # m/s at throttle = 1.0  (real JetBot ~ 0.3-0.5 m/s)
OMEGA_MAX = math.pi / 3 # rad/s at steer = 1.0   (~60 °/s)
DT = 0.10               # sim step (seconds)

# --- camera model ----------------------------------------------------------

HFOV = math.radians(58) # Astra Pro horizontal FOV
NUM_RAYS = 60           # 20 rays per zone
MIN_DEPTH_M = 0.60      # below this the real Astra reads as 0 (invalid)
MAX_DEPTH_M = 4.00      # we treat anything beyond as "clear"
RAY_STEP_M = 0.05

# --- bot footprint ---------------------------------------------------------

BOT_RADIUS_M = 0.15
TRAIL_LENGTH = 400

# --- view ------------------------------------------------------------------

DISPLAY_SCALE = 2.5     # pixels per occupancy cell
PANEL_W = 320

# --- starting pose (SW corner of the south corridor, facing east) ----------

START_X_M = 2.5
START_Y_M = 29.1
START_HEADING = 0.0


# --- obstacles to drop into the corridors for detection testing ------------

OBSTACLES = [
    # (x_m, y_m, size_m)
    (12.0, 29.1, 0.5),  # south corridor middle
    (22.5, 18.0, 0.5),  # east corridor middle
    (15.0,  1.5, 0.5),  # north corridor middle
    (1.5,  14.0, 0.5),  # west corridor middle
]


def add_obstacles(walls_grid):
    """Stamp each obstacle into a copy of the grid; return (combined, mask)
    where `mask` is a separate layer used only for visualization."""
    H, W = walls_grid.shape
    combined = walls_grid.copy()
    mask = np.zeros((H, W), dtype=np.uint8)
    for x_m, y_m, size_m in OBSTACLES:
        col = int(round(x_m / RESOLUTION_M))
        row = int(round(y_m / RESOLUTION_M))
        half = max(1, int(round(size_m / 2 / RESOLUTION_M)))
        r0, r1 = max(0, row - half), min(H, row + half)
        c0, c1 = max(0, col - half), min(W, col + half)
        combined[r0:r1, c0:c1] = WALL
        mask[r0:r1, c0:c1] = 1
    return combined, mask


# --- ray cast --------------------------------------------------------------

def cast_ray(grid, x_m, y_m, angle):
    H, W = grid.shape
    dx = math.cos(angle) * RAY_STEP_M
    dy = math.sin(angle) * RAY_STEP_M
    x, y = x_m, y_m
    d = 0.0
    while d < MAX_DEPTH_M:
        x += dx; y += dy; d += RAY_STEP_M
        c = int(x / RESOLUTION_M)
        r = int(y / RESOLUTION_M)
        if r < 0 or r >= H or c < 0 or c >= W or grid[r, c] == WALL:
            return d
    return MAX_DEPTH_M


def simulate_zones(grid, x_m, y_m, heading):
    """Cast NUM_RAYS evenly across the FOV, return (zones, rays).
    Distances < MIN_DEPTH_M are marked invalid (set to 0) to mimic the
    Astra Pro's behavior near its minimum range."""
    rays = np.empty(NUM_RAYS, dtype=np.float32)
    half = HFOV / 2
    for i in range(NUM_RAYS):
        angle = heading - half + (i + 0.5) / NUM_RAYS * HFOV
        rays[i] = cast_ray(grid, x_m, y_m, angle)
    valid_mask = rays >= MIN_DEPTH_M
    valid_mask[rays >= MAX_DEPTH_M - 1e-3] = True   # max-range counts as valid (treated as "clear")

    zw = NUM_RAYS // 3
    zones = []
    for i in range(3):
        s = i * zw
        e = (i + 1) * zw if i < 2 else NUM_RAYS
        zone_valid = rays[s:e][valid_mask[s:e]]
        if zone_valid.size < 3:
            zones.append(float("inf"))
        else:
            zones.append(float(np.percentile(zone_valid, 5)))
    return tuple(zones), rays


# --- bot dynamics ----------------------------------------------------------

def step_bot(state, decision):
    v = decision["throttle"] * V_MAX
    omega = decision["steer"] * OMEGA_MAX
    new_x = state["x"] + v * math.cos(state["theta"]) * DT
    new_y = state["y"] + v * math.sin(state["theta"]) * DT

    # Soft collision: don't let the bot center cross a wall.
    col = int(new_x / RESOLUTION_M)
    row = int(new_y / RESOLUTION_M)
    H, W = state["grid"].shape
    if 0 <= row < H and 0 <= col < W and state["grid"][row, col] != WALL:
        state["x"], state["y"] = new_x, new_y
    state["theta"] = (state["theta"] + omega * DT + math.pi) % (2 * math.pi) - math.pi

    state["trail"].append((state["x"], state["y"]))
    if len(state["trail"]) > TRAIL_LENGTH:
        state["trail"].pop(0)


# --- rendering -------------------------------------------------------------

def render_static_map(grid, mask):
    """Return a (H, W, 3) BGR array: walls dark grey, obstacles orange."""
    H, W = grid.shape
    out = np.full((H, W, 3), 240, dtype=np.uint8)
    out[grid == WALL] = (60, 60, 60)
    out[mask == 1] = (40, 110, 230)   # BGR orange/red
    return out


# --- main ------------------------------------------------------------------

def main():
    walls = build_grid()
    grid, mask = add_obstacles(walls)
    H, W = grid.shape
    map_w_px = int(W * DISPLAY_SCALE)
    map_h_px = int(H * DISPLAY_SCALE)

    root = tk.Tk()
    root.title("JetBot Depth-Nav Simulator — Light Eng 2F")
    canvas = tk.Canvas(
        root,
        width=map_w_px + PANEL_W,
        height=map_h_px,
        bg="#101010",
        highlightthickness=0,
    )
    canvas.pack()

    base_bgr = render_static_map(grid, mask)
    base_rgb = base_bgr[:, :, ::-1]
    base_pil = Image.fromarray(base_rgb).resize((map_w_px, map_h_px), Image.NEAREST)
    map_photo = ImageTk.PhotoImage(base_pil)
    canvas.create_image(0, 0, anchor="nw", image=map_photo, tags="map")

    nav = Navigator()
    state = {
        "x": START_X_M, "y": START_Y_M, "theta": START_HEADING,
        "trail": [], "running": True, "paused": False,
        "frame_no": 0, "grid": grid,
    }

    def reset(*_):
        state["x"] = START_X_M
        state["y"] = START_Y_M
        state["theta"] = START_HEADING
        state["trail"].clear()
        state["frame_no"] = 0
        nav.reset()

    def quit_(*_):
        state["running"] = False
        root.quit()

    def toggle_pause(*_):
        state["paused"] = not state["paused"]

    def teleport(event):
        if event.x > map_w_px:
            return
        state["x"] = event.x / DISPLAY_SCALE * RESOLUTION_M
        state["y"] = event.y / DISPLAY_SCALE * RESOLUTION_M
        state["trail"].clear()
        nav.reset()

    root.bind("<Escape>", quit_); root.bind("q", quit_); root.bind("Q", quit_)
    root.bind("r", reset);        root.bind("R", reset)
    root.bind("<space>", toggle_pause)
    canvas.bind("<Button-1>", teleport)
    root.protocol("WM_DELETE_WINDOW", quit_)

    def w2c(x_m, y_m):
        return (x_m / RESOLUTION_M * DISPLAY_SCALE,
                y_m / RESOLUTION_M * DISPLAY_SCALE)

    def update():
        if not state["running"]:
            return
        zones, rays = simulate_zones(grid, state["x"], state["y"], state["theta"])
        decision = nav.step(zones)
        if not state["paused"]:
            step_bot(state, decision)
            state["frame_no"] += 1

        canvas.delete("dynamic")

        # Trail
        if len(state["trail"]) > 1:
            pts = []
            for x_m, y_m in state["trail"]:
                cx, cy = w2c(x_m, y_m)
                pts.extend([cx, cy])
            canvas.create_line(*pts, fill="#5060d0", width=1, tags="dynamic")

        bx, by = w2c(state["x"], state["y"])

        # FOV cone — sample every 3 rays for a polygon outline
        cone_pts = [bx, by]
        for i in range(0, NUM_RAYS, 3):
            angle = state["theta"] - HFOV / 2 + (i + 0.5) / NUM_RAYS * HFOV
            d = float(rays[i])
            tx_m = state["x"] + math.cos(angle) * d
            ty_m = state["y"] + math.sin(angle) * d
            tx, ty = w2c(tx_m, ty_m)
            cone_pts.extend([tx, ty])
        canvas.create_polygon(
            *cone_pts,
            outline="#80ff80", fill="#1a3a1a", stipple="gray25",
            width=1, tags="dynamic",
        )

        # Zone-colored ray bundles for visual distance feedback
        zone_colors_by_status = {
            "STOP":    ("#ff5050", "#ff5050", "#ff5050"),
            "AVOID":   ("#ffaa30", "#ff5050", "#ffaa30"),
            "GO":      ("#60ff60", "#60ff60", "#60ff60"),
            "REVERSE": ("#a060ff", "#a060ff", "#a060ff"),
        }
        zw = NUM_RAYS // 3
        for zone_idx in range(3):
            s = zone_idx * zw
            e = (zone_idx + 1) * zw if zone_idx < 2 else NUM_RAYS
            d_val = zones[zone_idx]
            color = (
                "#60ff60" if d_val > DIST_CAUTION_M else
                "#ffaa30" if d_val > DIST_STOP_M else
                "#ff5050"
            )
            for i in (s, (s + e) // 2, e - 1):
                angle = state["theta"] - HFOV / 2 + (i + 0.5) / NUM_RAYS * HFOV
                d = float(rays[i])
                tx, ty = w2c(state["x"] + math.cos(angle) * d,
                              state["y"] + math.sin(angle) * d)
                canvas.create_line(bx, by, tx, ty, fill=color, width=1, tags="dynamic")

        # Bot circle + heading arrow
        r_px = BOT_RADIUS_M / RESOLUTION_M * DISPLAY_SCALE
        canvas.create_oval(bx - r_px, by - r_px, bx + r_px, by + r_px,
                           outline="#ffd060", fill="#403010", width=2, tags="dynamic")
        hx, hy = w2c(state["x"] + math.cos(state["theta"]) * 0.6,
                     state["y"] + math.sin(state["theta"]) * 0.6)
        canvas.create_line(bx, by, hx, hy, fill="#ffd060", width=3,
                           arrow="last", tags="dynamic")

        # Side panel
        px0 = map_w_px + 12
        py = 18
        line_h = 22

        def text(t, color="white", size=12, bold=False):
            nonlocal py
            font = ("DejaVu Sans Mono", size, "bold" if bold else "normal")
            canvas.create_text(px0, py, anchor="nw", text=t, fill=color, font=font, tags="dynamic")
            py += line_h

        status = decision["status"]
        status_color = {
            "STOP": "#ff5050", "AVOID": "#ffaa30",
            "GO": "#60ff60", "REVERSE": "#a060ff",
        }.get(status, "white")
        text(status, color=status_color, size=22, bold=True)
        py += 8
        text(f"throttle: {decision['throttle']:.2f}")
        text(f"steer:    {decision['steer']:+.2f}")
        py += 8
        for name, dist in zip(("L", "C", "R"), zones):
            d_color = (
                "#60ff60" if dist > DIST_CAUTION_M else
                "#ffaa30" if dist > DIST_STOP_M else
                "#ff5050"
            )
            d_str = f"{dist:.2f} m" if dist != float("inf") else "--"
            text(f"  {name}: {d_str}", color=d_color)
        py += 12
        text(f"x: {state['x']:5.2f} m", size=11)
        text(f"y: {state['y']:5.2f} m", size=11)
        text(f"θ: {math.degrees(state['theta']):+5.0f}°", size=11)
        text(f"frame: {state['frame_no']}", size=11)
        py += 16
        text("Space: pause", size=10, color="#909090")
        text("R: reset", size=10, color="#909090")
        text("Click map: teleport", size=10, color="#909090")
        text("Q/Esc: quit", size=10, color="#909090")

        if state["paused"]:
            py += 12
            text("[PAUSED]", color="#ffff60", bold=True)

        root.after(int(DT * 1000), update)

    root.after(0, update)
    try:
        root.mainloop()
    finally:
        state["running"] = False


if __name__ == "__main__":
    main()
