#!/usr/bin/env python3
"""Standalone, illustrative cone perception simulation (no ROS required)."""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_cones(path):
    positions, tags = [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"tag", "x", "y"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain tag,x,y columns.")
        for row in reader:
            positions.append([float(row["x"]), float(row["y"])])
            tags.append(row["tag"].strip().lower())
    points = np.asarray(positions, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("Cone coordinates must be finite.")
    return points, np.asarray(tags)


def resample(points, count, closed):
    """Sample a polyline uniformly by distance, removing repeated vertices."""
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if closed and np.linalg.norm(points[-1] - points[0]) < 1e-8:
        points = points[:-1]
    if len(points) < 2:
        raise ValueError("Each boundary needs at least two distinct cones.")
    if closed:
        points = np.vstack([points, points[0]])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0, distances[-1], count, endpoint=not closed)
    return np.column_stack([np.interp(samples, distances, points[:, axis]) for axis in (0, 1)])


def middle_line(points, tags, closed=True):
    boundaries = [points[tags == tag] for tag in ("blue", "yellow")]
    minimum = 3 if closed else 2
    if any(len(boundary) < minimum for boundary in boundaries):
        raise ValueError(f"Need at least {minimum} blue and {minimum} yellow cones, ordered along each boundary.")
    left, right = [resample(boundary, 800, closed) for boundary in boundaries]
    # Align direction and starting phase; CSV boundaries can start at different cones.
    candidates = []
    for oriented in (right, right[::-1]):
        shifts = range(len(right)) if closed else (0,)
        for shift in shifts:
            aligned = np.roll(oriented, shift, axis=0)
            candidates.append((np.mean(np.sum((left - aligned) ** 2, axis=1)), oriented, shift))
    _, oriented, shift = min(candidates, key=lambda item: item[0])
    return resample((left + np.roll(oriented, shift, axis=0)) / 2, 1600, closed)


def pose_at(line, distance, closed):
    path = np.vstack([line, line[0]]) if closed else line
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    distance = distance % cumulative[-1] if closed else np.clip(distance, 0, cumulative[-1])
    index = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
    fraction = (distance - cumulative[index]) / max(lengths[index], 1e-12)
    direction = path[index + 1] - path[index]
    return path[index] + fraction * direction, np.arctan2(direction[1], direction[0])


def scan(points, position, heading, max_range, decay_distance, rng):
    delta = points - position
    distances = np.linalg.norm(delta, axis=1)
    bearings = (np.arctan2(delta[:, 1], delta[:, 0]) - heading + np.pi) % (2 * np.pi) - np.pi
    visible = (distances <= max_range) & (np.abs(bearings) <= np.pi / 3)
    probability = np.where(visible, np.exp(-distances / decay_distance), 0.0)
    detected = visible & (rng.random(len(points)) < probability)
    return visible, detected, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path(__file__).with_name("sample_track.csv"))
    parser.add_argument("--open-track", action="store_true", help="Stop at the end instead of looping.")
    parser.add_argument("--speed", type=float, default=5.0, help="Car speed in m/s (default: 5).")
    parser.add_argument("--range", dest="max_range", type=float, default=30.0, help="Lidar range in metres.")
    parser.add_argument("--decay-distance", type=float, default=20.0, help="Distance where detection probability is exp(-1).")
    parser.add_argument("--hz", type=float, default=10.0, help="Independent lidar scans per second.")
    parser.add_argument("--duration", type=float, help="Simulation seconds; defaults to one lap/traversal.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="Run without opening a window.")
    parser.add_argument("--save", type=Path, help="Save final .png or animated .gif.")
    parser.add_argument("--log", type=Path, help="Write per-cone, per-scan ground truth and detection CSV.")
    args = parser.parse_args()
    for name in ("speed", "max_range", "decay_distance", "hz", "duration"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if args.save and args.save.suffix.lower() not in (".png", ".gif"):
        parser.error("--save must end in .png or .gif")
    try:
        points, tags = load_cones(args.csv)
        line = middle_line(points, tags, not args.open_track)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    closed = not args.open_track
    length = np.linalg.norm(np.diff(np.vstack([line, line[0]]) if closed else line, axis=0), axis=1).sum()
    duration = args.duration or length / args.speed
    if args.open_track:
        duration = min(duration, length / args.speed)
    times = np.minimum(np.arange(int(np.ceil(duration * args.hz)) + 1) / args.hz, duration)
    rng = np.random.default_rng(args.seed)
    frames = []
    for time in times:
        position, heading = pose_at(line, time * args.speed, closed)
        frames.append((position, heading, *scan(points, position, heading, args.max_range, args.decay_distance, rng)))
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "car_x", "car_y", "heading_rad", "cone_id", "tag", "x", "y", "in_fov_and_range", "probability", "detected"])
            for time, (position, heading, visible, detected, probability) in zip(times, frames):
                for index, point in enumerate(points):
                    writer.writerow([time, *position, heading, index, tags[index], *point, int(visible[index]), probability[index], int(detected[index])])
    opportunities = sum(frame[2].sum() for frame in frames)
    detections = sum(frame[3].sum() for frame in frames)
    print(f"{len(points)} cones | {length:.1f} m middle line | {len(frames)} scans | {detections}/{opportunities} eligible cones detected")
    if args.headless and not args.save:
        return

    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon, Wedge

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = [{"blue": "royalblue", "yellow": "gold", "orange": "darkorange", "orange_big": "darkorange"}.get(tag, "gray") for tag in tags]
    ax.scatter(*points.T, c=colors, marker="^", s=36, edgecolors="gray", alpha=0.45, label="True cones")
    drawn_line = np.vstack([line, line[0]]) if closed else line
    ax.plot(*drawn_line.T, "--", color="gray", linewidth=1, label="Middle line")
    eligible = ax.scatter([], [], s=95, facecolors="none", edgecolors="gray", label="Inside lidar view")
    hits = ax.scatter([], [], s=100, facecolors="none", edgecolors="limegreen", linewidths=2, label="Detected this scan")
    wedge = Wedge((0, 0), args.max_range, -60, 60, color="deepskyblue", alpha=0.12)
    ax.add_patch(wedge)
    car = Polygon(np.zeros((3, 2)), color="crimson", zorder=5)
    ax.add_patch(car)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="2D cone perception — 120° lidar")
    ax.set_aspect("equal")
    margin = 6
    ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    def update(index):
        position, heading, visible, detected, _ = frames[index]
        angle = np.degrees(heading)
        wedge.set_center(position)
        wedge.set_theta1(angle - 60)
        wedge.set_theta2(angle + 60)
        rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        car.set_xy(np.array([[1.3, 0], [-0.9, 0.65], [-0.9, -0.65]]) @ rotation.T + position)
        eligible.set_offsets(points[visible])
        hits.set_offsets(points[detected])
        status.set_text(f"t = {times[index]:5.1f} s   speed = {args.speed:g} m/s\nvisible: {visible.sum():3d}   detected: {detected.sum():3d}\np(d) = exp(-d / {args.decay_distance:g} m)")
        return wedge, car, eligible, hits, status

    animation = None
    if not args.headless or (args.save and args.save.suffix.lower() == ".gif"):
        animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.hz, repeat=False)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.suffix.lower() == ".gif":
            animation.save(args.save, writer=PillowWriter(fps=args.hz))
        else:
            update(len(frames) - 1)
            fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
    if not args.headless:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
