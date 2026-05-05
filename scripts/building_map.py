"""Light Engineering Building, Floor 2 — occupancy map.

Hand-traced from a sketch with measured outer dimensions and a constant
8 ft corridor width. The result is a 2D occupancy grid:

    cell value  0  = free space (corridor)
    cell value  255 = wall

The two interior blocks (rooms) sit inside a perimeter corridor and are
separated from each other by an 8 ft transit gap, giving the floor a
"figure-8" navigation topology.

Tweak the *_FT constants at the top to refine geometry without touching
any of the grid-building code.
"""
import numpy as np

# --- units ------------------------------------------------------------------

FT_TO_M = 0.3048
RESOLUTION_M = 0.10            # 10 cm per cell → an 8 ft corridor is ~24 cells across

# --- building dimensions (all in feet) -------------------------------------

OUTER_H_FT = 100               # building exterior, north-south
OUTER_W_FT = 80                # building exterior, east-west
CORRIDOR_FT = 8                # constant: outer corridor + transit gap between rooms

# Interior block heights (north-south). Total inner height available for
# rooms = OUTER_H_FT − 2*CORRIDOR_FT − CORRIDOR_FT(gap) = 100 − 24 = 76 ft.
# Eyeballed from sketch — adjust freely:
TOP_ROOM_H_FT = 50
BOT_ROOM_H_FT = 26
assert TOP_ROOM_H_FT + BOT_ROOM_H_FT + 3 * CORRIDOR_FT == OUTER_H_FT, (
    "room heights + corridors don't sum to outer height — fix the constants"
)

# --- grid encoding ---------------------------------------------------------

WALL = np.uint8(255)
FREE = np.uint8(0)


def ft_to_cells(ft: float) -> int:
    return int(round(ft * FT_TO_M / RESOLUTION_M))


def build_grid() -> np.ndarray:
    """Return the occupancy grid as an HxW uint8 array."""
    H = ft_to_cells(OUTER_H_FT)
    W = ft_to_cells(OUTER_W_FT)
    cor = ft_to_cells(CORRIDOR_FT)
    top_h = ft_to_cells(TOP_ROOM_H_FT)
    bot_h = ft_to_cells(BOT_ROOM_H_FT)

    grid = np.full((H, W), FREE, dtype=np.uint8)

    # Outer perimeter wall (1 cell thick)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    # Top interior room
    grid[cor:cor + top_h, cor:W - cor] = WALL

    # Bottom interior room (8 ft transit gap above it)
    bot_y0 = cor + top_h + cor
    grid[bot_y0:bot_y0 + bot_h, cor:W - cor] = WALL

    return grid


def grid_size_m() -> tuple:
    """Physical size of the grid in metres (height_m, width_m)."""
    return (OUTER_H_FT * FT_TO_M, OUTER_W_FT * FT_TO_M)


def cell_to_world_m(row: int, col: int) -> tuple:
    """Convert a (row, col) grid cell to (x_m, y_m) world coordinates.
    Origin is the top-left of the grid; +x is east (right), +y is south (down)."""
    return (col * RESOLUTION_M, row * RESOLUTION_M)


def world_m_to_cell(x_m: float, y_m: float) -> tuple:
    """Inverse of cell_to_world_m. Returns (row, col)."""
    return (int(round(y_m / RESOLUTION_M)), int(round(x_m / RESOLUTION_M)))


if __name__ == "__main__":
    import cv2

    grid = build_grid()
    H, W = grid.shape
    h_m, w_m = grid_size_m()
    print(f"grid: {H} x {W} cells")
    print(f"physical: {h_m:.2f} m N-S × {w_m:.2f} m E-W")
    print(f"resolution: {RESOLUTION_M} m/cell ({RESOLUTION_M * 100:.0f} cm)")
    print(f"corridor width: {ft_to_cells(CORRIDOR_FT)} cells "
          f"({CORRIDOR_FT * FT_TO_M:.2f} m)")

    # Render: walls black, free white, plus a faint metre grid + scale bar
    out = np.full((H, W, 3), 255, dtype=np.uint8)
    out[grid == WALL] = (0, 0, 0)

    # 1-metre faint gridlines
    step = int(round(1.0 / RESOLUTION_M))
    for r in range(0, H, step):
        out[r, :] = np.where(out[r, :] == 255, 230, out[r, :])
    for c in range(0, W, step):
        out[:, c] = np.where(out[:, c] == 255, 230, out[:, c])

    # 5 m scale bar bottom-left
    bar_len = 5 * step
    cv2.line(out, (10, H - 20), (10 + bar_len, H - 20), (0, 0, 200), 3)
    cv2.putText(out, "5 m", (10, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    out_path = "/home/cad281/jetbot-project/building_map.png"
    cv2.imwrite(out_path, out)
    print(f"saved {out_path}")
