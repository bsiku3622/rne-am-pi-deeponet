"""Compare a trained DeepONeuralNet against the simulation it was fitted to.

Draws one row per requested time, three panels wide: the finite-element field,
the network prediction, and their signed difference. Truth and prediction share
a colour scale so they can be read against each other; the error panel uses a
symmetric diverging scale centred on zero, so blue is under-prediction and red
is over-prediction.

Two slices are available. ``--plane top`` shows the ``z = z_max`` surface the
laser scans, over ``(x, y)``. ``--plane track`` cuts along the scan line at
``y = y_c`` and shows ``(x, z)``, which is where the melt pool depth lives and
where the model is hardest to fit.

Examples::

    python visualize.py --checkpoint checkpoint.pt --power 500
    python visualize.py --power 700 --plane track --times 0.5 1.5 2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from calibrate import load_grid, Grid
from model import DeepONeuralNet

MM = 1.0e-3


def find_grid(data_dir: Path, power: float) -> Grid:
    """Load the simulation file whose laser power matches ``power``."""
    paths = sorted(data_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy files under {data_dir}")

    available = []
    for path in paths:
        grid = load_grid(path)
        if abs(grid.power - power) < 1e-9:
            return grid
        available.append(grid.power)
    raise ValueError(f"no file at P = {power} W; available: {sorted(available)}")


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[DeepONeuralNet, dict]:
    """Rebuild the network from the architecture stored alongside the weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture is None:
        raise KeyError(
            f"{checkpoint_path} predates the `architecture` key; retrain or add it by hand"
        )

    model = DeepONeuralNet(**architecture)
    model.load_state_dict(checkpoint["model"])  # normalisation buffers ride along
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(
    model: DeepONeuralNet,
    coords: np.ndarray,
    power: float,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    """Evaluate the model on ``[N, 4]`` physical coordinates, returning ``[N]`` Kelvin."""
    dtype = next(model.parameters()).dtype
    tensor = torch.as_tensor(coords, dtype=dtype, device=device)
    outputs = []
    for start in range(0, tensor.size(0), chunk):
        block = tensor[start : start + chunk]
        laser_power = torch.full((block.size(0), 1), power, dtype=dtype, device=device)
        outputs.append(model(laser_power, block).squeeze(-1).cpu())
    return torch.cat(outputs).numpy()


def slice_coords(
    grid: Grid, plane: str, time: float, track_y: float
) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Return ``(coords[N,4], truth[a,b], extent_mm, axis_labels)`` for one slice."""
    time_index = int(np.argmin(np.abs(grid.t - time)))
    actual_time = float(grid.t[time_index])

    if plane == "top":
        first, second = grid.x, grid.y
        a, b = np.meshgrid(first, second, indexing="ij")
        c = np.full(a.shape, grid.z[-1])
        truth = grid.temperature[:, :, -1, time_index]
        coords = np.stack([a, b, c, np.full(a.shape, actual_time)], axis=-1)
        labels = ("x [mm]", "y [mm]")
    elif plane == "track":
        row = int(np.argmin(np.abs(grid.y - track_y)))
        first, second = grid.x, grid.z
        a, c = np.meshgrid(first, second, indexing="ij")
        b = np.full(a.shape, grid.y[row])
        truth = grid.temperature[:, row, :, time_index]
        coords = np.stack([a, b, c, np.full(a.shape, actual_time)], axis=-1)
        labels = ("x [mm]", "z [mm]")
    else:
        raise ValueError(f"unknown plane {plane!r}; expected 'top' or 'track'")

    extent = (
        float(first[0]) / MM,
        float(first[-1]) / MM,
        float(second[0]) / MM,
        float(second[-1]) / MM,
    )
    return coords.reshape(-1, 4), truth, extent, labels


def draw(
    grid: Grid,
    model: DeepONeuralNet,
    times: list[float],
    plane: str,
    track_y: float,
    device: torch.device,
) -> Figure:
    """Render the truth / prediction / error grid and print per-slice metrics."""
    # Panels use an equal aspect, so let the slice's own shape set the row height
    # instead of leaving a band of whitespace above and below each strip.
    _, _, probe_extent, _ = slice_coords(grid, plane, times[0], track_y)
    span_x = probe_extent[1] - probe_extent[0]
    span_y = probe_extent[3] - probe_extent[2]
    panel_width = 4.3
    row_height = panel_width * (span_y / span_x) + 1.5

    figure, axes = plt.subplots(
        len(times),
        3,
        figsize=(15, row_height * len(times)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"P = {grid.power:.0f} W, plane = {plane}", fontsize=13)

    for row, time in enumerate(times):
        coords, truth, extent, labels = slice_coords(grid, plane, time, track_y)
        prediction = predict(model, coords, grid.power, device).reshape(truth.shape)
        error = prediction - truth

        rmse = float(np.sqrt((error**2).mean()))
        worst = float(np.abs(error).max())
        print(
            f"  t = {time:4.2f}s  RMSE = {rmse:8.3f} K   max |error| = {worst:9.3f} K"
        )

        # Truth and prediction share limits; the error scale is symmetric about zero.
        low, high = float(truth.min()), float(truth.max())
        bound = max(float(np.abs(error).max()), 1e-9)

        style = dict(origin="lower", extent=extent, aspect="equal")
        field_style = dict(vmin=low, vmax=high, cmap="inferno")

        truth_image = axes[row][0].imshow(truth.T, **style, **field_style)
        axes[row][1].imshow(prediction.T, **style, **field_style)
        error_image = axes[row][2].imshow(
            error.T, **style, vmin=-bound, vmax=bound, cmap="RdBu_r"
        )

        axes[row][0].set_title(f"simulation\nt = {time:.2f} s", fontsize=10)
        axes[row][1].set_title(f"DeepONeuralNet\nRMSE {rmse:.1f} K", fontsize=10)
        axes[row][2].set_title(
            f"prediction - simulation\nmax |error| {worst:.0f} K", fontsize=10
        )

        for column in range(3):
            axes[row][column].set_xlabel(labels[0])
        axes[row][0].set_ylabel(labels[1])

        # Truth and prediction share a scale, so one bar serves both.
        figure.colorbar(truth_image, ax=axes[row][:2].tolist(), label="K", shrink=0.9)
        figure.colorbar(error_image, ax=axes[row][2], label="K", shrink=0.9)

    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--power", type=float, required=True, help="which simulation to compare against"
    )
    parser.add_argument("--plane", choices=("top", "track"), default="top")
    parser.add_argument("--times", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    parser.add_argument(
        "--track-y",
        type=float,
        default=4.9922,
        help="scan-line y in mm, for --plane track",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="save here instead of showing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    grid = find_grid(args.data_dir, args.power)
    model, checkpoint = load_model(args.checkpoint, device)
    print(
        f"[load] {args.checkpoint.name}: iteration {checkpoint['iteration']}, "
        f"val RMSE {checkpoint['val_rmse']:.3f} K"
    )
    print(f"[load] {grid.name}: {grid.temperature.shape} at P = {grid.power} W")

    figure = draw(grid, model, args.times, args.plane, args.track_y * MM, device)

    if args.out is None:
        plt.show()
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.out, dpi=140)
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
