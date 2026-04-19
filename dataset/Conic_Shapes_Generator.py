from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(".")
SAMPLE_COUNT = 6
SAMPLES_PER_SHAPE = 1000


def plotter(x_data, y_data, title, output_dir: Path = OUTPUT_DIR):
    fig = plt.figure(figsize=[10, 10])
    plt.plot(x_data, y_data, "b--")
    plt.xlabel("X-axis", fontsize=14)
    plt.ylabel("Y-axis", fontsize=14)
    plt.ylim(-18, 18)
    plt.xlim(-18, 18)
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.grid(True)
    save_file = output_dir / f"{title}.svg"
    plt.savefig(save_file)
    plt.close(fig)


def rotateCoordinates(x_data, y_data, rot_angle):
    x_ = x_data * math.cos(rot_angle) - y_data * math.sin(rot_angle)
    y_ = x_data * math.sin(rot_angle) + y_data * math.cos(rot_angle)
    return x_, y_


def get_n_samples(x_data, y_data, n):
    indexes = np.round(np.linspace(0, len(x_data) - 1, n)).astype(int)
    return x_data[indexes], y_data[indexes]


# returns a single random index from an array

def get_random_index(array_size, rng: np.random.Generator):
    return int(rng.integers(0, array_size))


def build_dataset(x_, y_, shape):
    row = {}
    for i in range(len(x_)):
        row[f"x{i + 1}"] = x_[i]
        row[f"y{i + 1}"] = y_[i]
    row["shape"] = shape
    return row


def createParabola(focal_length, centre, rotation):
    t = np.linspace(-math.pi, math.pi, 100)
    x_parabola = focal_length * t**2
    y_parabola = 2 * focal_length * t
    if rotation is not None:
        x_parabola, y_parabola = rotateCoordinates(x_parabola, y_parabola, rotation)
    x_parabola = x_parabola + centre[0]
    y_parabola = y_parabola + centre[1]
    return x_parabola, y_parabola


def createCircle(radius, centre):
    theta = np.linspace(0, 2 * math.pi, 100)
    x_circle = radius * np.cos(theta) + centre[0]
    y_circle = radius * np.sin(theta) + centre[1]
    return x_circle, y_circle


def createEllipse(major_axis, minor_axis, centre, rotation):
    theta = np.linspace(0, 2 * math.pi, 100)
    x_ellipse = major_axis * np.cos(theta)
    y_ellipse = minor_axis * np.sin(theta)
    if rotation is not None:
        x_ellipse, y_ellipse = rotateCoordinates(x_ellipse, y_ellipse, rotation)
    x_ellipse = x_ellipse + centre[0]
    y_ellipse = y_ellipse + centre[1]
    return x_ellipse, y_ellipse


def createHyperbola(major_axis, conjugate_axis, centre, rotation):
    theta = np.linspace(0, 2 * math.pi, 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_hyperbola = major_axis * (1 / np.cos(theta))
        y_hyperbola = conjugate_axis * np.tan(theta)
    if rotation is not None:
        x_hyperbola, y_hyperbola = rotateCoordinates(x_hyperbola, y_hyperbola, rotation)
    x_hyperbola = x_hyperbola + centre[0]
    y_hyperbola = y_hyperbola + centre[1]
    return x_hyperbola, y_hyperbola


def preview_examples(output_dir: Path = OUTPUT_DIR):
    angle = [0, math.pi / 4, math.pi * 2 / 3, math.pi * 4 / 3]

    j = 0
    for i in angle:
        j += 1
        x_parabola, y_parabola = createParabola(focal_length=1.8, centre=[-3 + j, -4 + j], rotation=i)
        plotter(x_parabola, y_parabola, f"Parabola {j}", output_dir=output_dir)

    centres = [[0, 0], [-1, -2], [2, -1.5], [-1.8, 1.2]]
    for i in centres:
        x_circle, y_circle = createCircle(centre=i, radius=15)
        plotter(x_circle, y_circle, f"Circle_{i[0]}_{i[1]}", output_dir=output_dir)

    j = 0
    for i in angle:
        j += 1
        x_ellipse, y_ellipse = createEllipse(major_axis=16, minor_axis=8, centre=[-1 + j, -1.5 + j], rotation=i)
        plotter(x_ellipse, y_ellipse, f"Ellipse {j}", output_dir=output_dir)

    j = 0
    for i in angle:
        j += 1
        x_hyperbola, y_hyperbola = createHyperbola(major_axis=5, conjugate_axis=3, centre=[-2 + j, 0 + j], rotation=i)
        plotter(x_hyperbola, y_hyperbola, f"Hyperbola {j}", output_dir=output_dir)



def _generate_shape_dataset(shape_name: str, samples_per_shape: int, sample_count: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    if shape_name == "parabola":
        focal_length_array = np.linspace(1, 20, 100)
        centre_x_arr = np.linspace(-12, 12, 100)
        centre_y_arr = np.linspace(-12, 12, 100)
        rotation_array = np.linspace(0, 2 * math.pi, 100)

        for _ in range(samples_per_shape):
            focal_length = focal_length_array[get_random_index(len(focal_length_array), rng)]
            centre_x = centre_x_arr[get_random_index(len(centre_x_arr), rng)]
            centre_y = centre_y_arr[get_random_index(len(centre_y_arr), rng)]
            rotation = rotation_array[get_random_index(len(rotation_array), rng)]
            x, y = createParabola(focal_length=focal_length, centre=[centre_x, centre_y], rotation=rotation)
            x_, y_ = get_n_samples(x, y, sample_count)
            rows.append(build_dataset(x_, y_, "parabola"))

    elif shape_name == "ellipse":
        major_axis_array = np.linspace(1, 20, 100)
        minor_axis_array = np.linspace(1, 20, 100)
        centre_x_arr = np.linspace(-12, 12, 100)
        centre_y_arr = np.linspace(-12, 12, 100)
        rotation_array = np.linspace(0, 2 * math.pi, 100)

        for _ in range(samples_per_shape):
            major_axis = major_axis_array[get_random_index(len(major_axis_array), rng)]
            minor_axis = minor_axis_array[get_random_index(len(minor_axis_array), rng)]
            centre_x = centre_x_arr[get_random_index(len(centre_x_arr), rng)]
            centre_y = centre_y_arr[get_random_index(len(centre_y_arr), rng)]
            rotation = rotation_array[get_random_index(len(rotation_array), rng)]
            x, y = createEllipse(major_axis=major_axis, minor_axis=minor_axis, centre=[centre_x, centre_y], rotation=rotation)
            x_, y_ = get_n_samples(x, y, sample_count)
            rows.append(build_dataset(x_, y_, "ellipse"))

    elif shape_name == "hyperbola":
        major_axis_array = np.linspace(1, 20, 100)
        conjugate_axis_array = np.linspace(1, 20, 100)
        centre_x_arr = np.linspace(-12, 12, 100)
        centre_y_arr = np.linspace(-12, 12, 100)
        rotation_array = np.linspace(0, 2 * math.pi, 100)

        for _ in range(samples_per_shape):
            major_axis = major_axis_array[get_random_index(len(major_axis_array), rng)]
            conjugate_axis = conjugate_axis_array[get_random_index(len(conjugate_axis_array), rng)]
            centre_x = centre_x_arr[get_random_index(len(centre_x_arr), rng)]
            centre_y = centre_y_arr[get_random_index(len(centre_y_arr), rng)]
            rotation = rotation_array[get_random_index(len(rotation_array), rng)]
            x, y = createHyperbola(major_axis=major_axis, conjugate_axis=conjugate_axis, centre=[centre_x, centre_y], rotation=rotation)
            x_, y_ = get_n_samples(x, y, sample_count)
            rows.append(build_dataset(x_, y_, "hyperbola"))

    elif shape_name == "circle":
        radius_array = np.linspace(1, 20, 100)
        centre_x_arr = np.linspace(-12, 12, 100)
        centre_y_arr = np.linspace(-12, 12, 100)

        for _ in range(samples_per_shape):
            radius = radius_array[get_random_index(len(radius_array), rng)]
            centre_x = centre_x_arr[get_random_index(len(centre_x_arr), rng)]
            centre_y = centre_y_arr[get_random_index(len(centre_y_arr), rng)]
            x, y = createCircle(radius=radius, centre=[centre_x, centre_y])
            x_, y_ = get_n_samples(x, y, sample_count)
            rows.append(build_dataset(x_, y_, "circle"))

    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    return pd.DataFrame(rows)


def generate_combined_dataset(
    samples_per_shape: int = SAMPLES_PER_SHAPE,
    sample_count: int = SAMPLE_COUNT,
    save_path: str | Path = "Conic-Section_dataset.csv",
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    parabola_dataset = _generate_shape_dataset("parabola", samples_per_shape, sample_count, rng)
    ellipse_dataset = _generate_shape_dataset("ellipse", samples_per_shape, sample_count, rng)
    hyperbola_dataset = _generate_shape_dataset("hyperbola", samples_per_shape, sample_count, rng)
    circle_dataset = _generate_shape_dataset("circle", samples_per_shape, sample_count, rng)

    combined_dataset = pd.concat(
        [parabola_dataset, ellipse_dataset, hyperbola_dataset, circle_dataset],
        ignore_index=True,
    )
    combined_dataset.to_csv(save_path, index=False)
    return combined_dataset


if __name__ == "__main__":
    output_csv = Path("Conic-Section_dataset.csv")
    dataset = generate_combined_dataset(save_path=output_csv, seed=42)
    print(f"Dataset creato con {len(dataset)} righe: {output_csv.resolve()}")
