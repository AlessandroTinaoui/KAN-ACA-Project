"""Run the continual-learning experiment with the class order reversed.

This file keeps the main implementation in continual_conic.py and changes only
the schedule/output names. It is meant to compare whether learning the same
classes in a different order changes accuracy, loss, and forgetting.
"""

from __future__ import annotations

import sys

import continual_conic as experiment


# Same structure as the main experiment, but the class order is reversed:
# circle + hyperbola first, then ellipse, then parabola, then balanced review.
experiment.TRAINING_SCHEDULE = [
    {"circle": 800, "hyperbola": 800},
    {"circle": 300, "hyperbola": 300, "ellipse": 800},
    {"circle": 200, "hyperbola": 200, "ellipse": 200, "parabola": 800},
    {"circle": 200, "hyperbola": 200, "ellipse": 200, "parabola": 200},
]


# Use separate outputs, so the reversed run does not overwrite the normal run.
experiment.METRICS_OUT = "continual_metrics_reversed.csv"
experiment.CLASS_TESTS_OUT = "continual_class_tests_reversed.csv"
experiment.ACCURACY_CONFIG_OUT = "continual_accuracy_config_reversed.json"
experiment.RUN_CONFIG_OUT = "continual_run_config_reversed.json"
experiment.FINAL_MODEL_OUT = "continual_model_final_reversed.pt"
experiment.PLOT_DIR = "continual_plots_reversed"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script does not accept command-line arguments: edit this file."
        )

    experiment.train_continual()
