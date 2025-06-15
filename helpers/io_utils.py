import os
import datetime

def setup_directories(output_dir: str = "output") -> str:
    """Create a unique output directory inside the given parent directory,
        with subdirectories for final models and metrics."""
    # Ensure the parent output directory exists
    parent_dir = os.path.abspath(output_dir)
    os.makedirs(parent_dir, exist_ok=True)

    # Create a unique directory name using timestamp inside the parent directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_dir = os.path.join(parent_dir, f"run_{timestamp}")

    # Create subdirectories
    final_models_path = os.path.join(unique_output_dir, "final_models")
    metrics_path = os.path.join(unique_output_dir, "metrics")

    os.makedirs(final_models_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return unique_output_dir