"""
SPO (Spinning Parallelogram Operator) for Light Field Depth Estimation.
"""

import json
import os

from spo import SPO


DEFAULT_CONFIG = {
    "scale": 1.0,
    "bins": 64,
    "nD": 64,
    "sigma": 0.3,
    "guided_filter_radius": 10,
    "guided_filter_eps": 0.0001,
    "use_gpu": True,
}


def load_runtime_config(config_path):
    """Load global runtime parameters from config.json."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        user_config = json.load(config_file)

    config = DEFAULT_CONFIG.copy()
    config.update(user_config)

    int_keys = ("bins", "nD", "guided_filter_radius")
    for key in int_keys:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"`{key}` must be a positive integer, got {value!r}")

    if config["nD"] <= 1:
        raise ValueError(f"`nD` must be greater than 1, got {config['nD']!r}")

    positive_float_keys = ("scale", "sigma", "guided_filter_eps")
    for key in positive_float_keys:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"`{key}` must be a positive number, got {value!r}")
        config[key] = float(value)

    if not isinstance(config["use_gpu"], bool):
        raise ValueError(f"`use_gpu` must be a boolean, got {config['use_gpu']!r}")

    return config


def main():
    """Main function to run the SPO algorithm."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.json")
    filepath_input = os.path.join(base_dir, "input", "boxes")
    filepath_output = os.path.join(base_dir, "result", "boxes")

    config = load_runtime_config(config_path)

    os.makedirs(filepath_output, exist_ok=True)

    print("Starting SPO depth estimation...")
    SPO(
        filepath_input,
        filepath_output,
        config["scale"],
        config["bins"],
        config["nD"],
        config["sigma"],
        guided_filter_radius=config["guided_filter_radius"],
        guided_filter_eps=config["guided_filter_eps"],
        use_gpu=config["use_gpu"],
    )
    print("Depth estimation completed!")


if __name__ == "__main__":
    main()
