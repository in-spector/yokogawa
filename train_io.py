"""
I/O helpers for training scripts.
"""
import os


def prepare_run_dir(output_dir: str, run_id: str, tensorboard_dir: str) -> str:
    """Create standard run directory layout and return run_dir."""
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, tensorboard_dir), exist_ok=True)
    return run_dir


def write_latest_run(output_dir: str, latest_filename: str, run_dir: str):
    """Write path to the latest run file under output_dir."""
    latest_run_file = os.path.join(output_dir, latest_filename)
    with open(latest_run_file, "w") as f:
        f.write(run_dir + "\n")

