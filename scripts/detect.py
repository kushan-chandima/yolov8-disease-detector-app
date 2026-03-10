"""
YOLOv8 Chili Disease Detection Module
======================================
Detects leaf curl disease in chili plant images using YOLOv8
and sends spray commands to ESP32 via serial communication.
"""

from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import serial
import time
import yaml
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. Defaults to config/config.yaml
                     relative to the project root.
    Returns:
        dict with configuration values.
    """
    if config_path is None:
        # Resolve project root (one level up from scripts/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# YOLODetector Class
# ---------------------------------------------------------------------------

class YOLODetector:
    """YOLOv8-based chili disease detector with ESP32 spray control."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.6,
                 color_map: dict = None, quadrant_config: dict = None):
        """
        Args:
            model_path: Path to YOLOv8 .pt weights file.
            confidence_threshold: Minimum detection confidence (0-1).
            color_map: Dict mapping class_id -> (B, G, R) colour tuple.
            quadrant_config: Dict with 'x_split' and 'y_split' pixel values.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Default colour map (BGR)
        self.color_map = color_map or {
            0: (255, 0, 0),   # Class 0 - Red
            1: (0, 255, 0),   # Class 1 - Green
        }

        # Quadrant split configuration
        self.quadrant = quadrant_config or {"x_split": 180, "y_split": 320}

        # Internal state (reset on each load_image call)
        self.image = None
        self.image_all_boxes = None
        self.image_top_box = None
        self.centers: list[tuple[int, int]] = []
        self.mean_center: tuple[int, int] | None = None
        self.best_box: tuple | None = None

        logger.info(
            "YOLODetector initialised  |  model=%s  confidence=%.2f",
            model_path, confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------

    def load_image(self, image_path: str) -> None:
        """Load an image from disk and prepare working copies.

        Args:
            image_path: Path to the input image file.
        Raises:
            ValueError: If the image cannot be read.
        """
        self.image = cv.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.image_all_boxes = self.image.copy()
        self.image_top_box = self.image.copy()

        # Reset detection state
        self.centers = []
        self.mean_center = None
        self.best_box = None

        logger.info("Image loaded: %s  (%dx%d)", image_path,
                     self.image.shape[1], self.image.shape[0])

    # ------------------------------------------------------------------
    # Object Detection
    # ------------------------------------------------------------------

    def detect_objects(self) -> None:
        """Run YOLOv8 inference and draw results on working copies."""
        if self.image is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        predictions = self.model.predict(self.image, verbose=False)
        best_score = 0.0

        for result in predictions:
            for cls, score, box in zip(
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
            ):
                if score < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.centers.append((cx, cy))

                class_name = result.names.get(int(cls), f"Class {int(cls)}")
                colour = self.color_map.get(int(cls), (0, 255, 255))

                # Draw bounding box on all-boxes image
                cv.rectangle(self.image_all_boxes, (x1, y1), (x2, y2), colour, 2)
                cv.circle(self.image_all_boxes, (cx, cy), 5, (0, 0, 255), -1)

                label = f"{class_name} {score:.2f}"
                cv.putText(
                    self.image_all_boxes, label, (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                )

                # Track the most confident detection
                if score > best_score:
                    best_score = score
                    self.best_box = (x1, y1, x2, y2, cx, cy)

        # Compute mean centre of all detections
        if self.centers:
            mean_cx = int(np.mean([c[0] for c in self.centers]))
            mean_cy = int(np.mean([c[1] for c in self.centers]))
            self.mean_center = (mean_cx, mean_cy)

            cv.circle(self.image_all_boxes, (mean_cx, mean_cy), 7, (0, 255, 255), -1)
            logger.info("Mean centre: (%d, %d)  |  %d detection(s)",
                        mean_cx, mean_cy, len(self.centers))
        else:
            logger.info("No detections above confidence threshold %.2f",
                        self.confidence_threshold)

        # Draw highest-confidence box on top-box image
        if self.best_box:
            x1, y1, x2, y2, best_cx, best_cy = self.best_box
            cv.rectangle(self.image_top_box, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv.circle(self.image_top_box, (best_cx, best_cy), 5, (0, 0, 255), -1)

    # ------------------------------------------------------------------
    # Quadrant / Area Determination
    # ------------------------------------------------------------------

    def determine_area(self) -> list[int]:
        """Determine the spray quadrant based on the mean detection centre.

        Quadrant layout (based on configurable split point):
            ┌───────┬───────┐
            │   1   │   2   │
            ├───────┼───────┤
            │   3   │   4   │
            └───────┴───────┘

        Returns:
            A list containing the quadrant number, or an empty list if
            no valid detections exist.
        """
        if not self.mean_center:
            return []

        mean_cx, mean_cy = self.mean_center
        x_split = self.quadrant["x_split"]
        y_split = self.quadrant["y_split"]

        if mean_cx < x_split and mean_cy < y_split:
            area = 1
        elif mean_cx >= x_split and mean_cy < y_split:
            area = 2
        elif mean_cx < x_split and mean_cy >= y_split:
            area = 3
        else:
            area = 4

        logger.info("Determined spray area: %d", area)
        return [area]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display_image(self, image_type: str = "original",
                      auto_close_delay: float = 0) -> None:
        """Display an image using matplotlib.

        Args:
            image_type: One of 'original', 'all_boxes', or 'top_box'.
            auto_close_delay: Seconds before auto-closing (0 = manual close).
        """
        images = {
            "original": self.image,
            "all_boxes": self.image_all_boxes,
            "top_box": self.image_top_box,
        }

        img = images.get(image_type)
        if img is None:
            raise ValueError(
                f"Invalid image_type '{image_type}'. "
                f"Choose from: {list(images.keys())}"
            )

        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Detection Result – {image_type}")
        plt.show(block=False)

        if auto_close_delay > 0:
            plt.pause(auto_close_delay)
            plt.close()

    # ------------------------------------------------------------------
    # ESP32 Communication
    # ------------------------------------------------------------------

    def send_to_esp32(self, area: list[int],
                      port: str = "COM3", baud_rate: int = 115200,
                      timeout: int = 2, connection_delay: int = 2) -> bool:
        """Send the spray area command to the ESP32 via serial.

        Args:
            area: List containing the quadrant number to spray.
            port: Serial port name (e.g. 'COM3').
            baud_rate: Serial baud rate.
            timeout: Serial read timeout in seconds.
            connection_delay: Seconds to wait after opening port.
        Returns:
            True if the command was sent successfully, False otherwise.
        """
        if not area:
            logger.warning("No area to send – skipping ESP32 command.")
            return False

        try:
            ser = serial.Serial(port, baud_rate, timeout=timeout)
            time.sleep(connection_delay)

            command = str(area[0])
            ser.write(command.encode())
            logger.info("Sent to ESP32 on %s: %s", port, command)
            ser.close()
            return True

        except serial.SerialException as e:
            logger.error("Serial communication error: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error communicating with ESP32: %s", e)
            return False


# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    model_cfg = config["model"]
    serial_cfg = config["serial"]
    detect_cfg = config["detection"]

    # Build colour map from config (convert lists to tuples)
    color_map = {
        int(k): tuple(v)
        for k, v in detect_cfg.get("color_map", {}).items()
    }

    # Initialise detector
    detector = YOLODetector(
        model_path=model_cfg["weights_path"],
        confidence_threshold=model_cfg["confidence_threshold"],
        color_map=color_map,
        quadrant_config=detect_cfg.get("quadrant", {}),
    )

    # ---- Choose your image path here ----
    image_path = input("Enter image path for detection: ").strip()

    # Load & detect
    detector.load_image(image_path)
    detector.detect_objects()

    # Display results
    print("\nDisplaying detection results...")
    detector.display_image(image_type="all_boxes", auto_close_delay=5)

    # Determine spray area
    area = detector.determine_area()
    print(f"Determined Area: {area}")

    # Send command to ESP32
    if area:
        detector.send_to_esp32(
            area,
            port=serial_cfg["port"],
            baud_rate=serial_cfg["baud_rate"],
            timeout=serial_cfg["timeout"],
            connection_delay=serial_cfg["connection_delay"],
        )
    else:
        print("No disease detected – no spray command sent.")
