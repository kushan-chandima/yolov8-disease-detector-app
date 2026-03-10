"""
YOLOv8 Chili Disease Model Training Script
============================================
Fine-tunes a YOLOv8 model on a custom chili disease dataset.
Configuration is loaded from config/config.yaml.
"""

from ultralytics import YOLO
import yaml
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load project configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to config/config.yaml
                     relative to the project root.
    Returns:
        dict with configuration values.
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


def create_dataset_yaml(dataset_cfg: dict, output_path: Path) -> Path:
    """Create the data.yaml required by YOLOv8 for training.

    Args:
        dataset_cfg: Dataset configuration dict from config.yaml.
        output_path: Directory where data.yaml will be written.
    Returns:
        Path to the created data.yaml file.
    """
    data_yaml = {
        "path": str(Path(dataset_cfg["path"]).resolve()),
        "train": dataset_cfg["train"],
        "val": dataset_cfg["val"],
        "nc": dataset_cfg["num_classes"],
        "names": dataset_cfg["class_names"],
    }

    os.makedirs(output_path, exist_ok=True)
    yaml_path = output_path / "data.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info("Dataset YAML created at %s", yaml_path)
    return yaml_path


def train_model(config: dict) -> None:
    """Train a YOLOv8 model using settings from configuration.

    Args:
        config: Full project configuration dict.
    """
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / dataset_cfg["path"]

    # Create dataset configuration YAML
    data_yaml_path = create_dataset_yaml(dataset_cfg, dataset_path)

    # Create training output directory
    train_output = project_root / train_cfg["project"]
    os.makedirs(train_output, exist_ok=True)

    # Load pretrained YOLOv8 model
    logger.info("Loading base model: %s", train_cfg["base_model"])
    model = YOLO(train_cfg["base_model"])

    # Start training
    logger.info(
        "Starting training  |  epochs=%d  imgsz=%d  project=%s  name=%s",
        train_cfg["epochs"],
        train_cfg["image_size"],
        train_output,
        train_cfg["experiment_name"],
    )

    model.train(
        data=str(data_yaml_path),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["image_size"],
        project=str(train_output),
        name=train_cfg["experiment_name"],
    )

    logger.info("Training complete! Results saved to %s", train_output)


# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    config = load_config()
    train_model(config)
