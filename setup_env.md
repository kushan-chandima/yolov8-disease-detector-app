# Environment Setup Guide

## Prerequisites

- **Python 3.10.8** (recommended — other 3.10.x versions should also work)
- **pip** (comes with Python)
- **Arduino IDE** (for ESP32 firmware upload)

## Python Environment Setup

### Option 1: Create a New Virtual Environment (Recommended)

```bash
# Navigate to the project directory
cd Yolov8-disease-detector-app

# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Option 2: Use the Existing FYP Environment

If you already have the environment at `C:\python project\virtual_env\fyp_env`:

```bash
# Activate the existing environment
"C:\python project\virtual_env\fyp_env\Scripts\activate"
```

> **Note:** The existing environment contains 97 packages. Only the 13 packages listed in `requirements.txt` are required by this project.

## Required Packages Summary

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | 8.3.105 | YOLOv8 model training & inference |
| torch | 2.6.0 | PyTorch deep learning backend |
| torchvision | 0.21.0 | PyTorch vision utilities |
| opencv-python | 4.11.0.86 | Image loading, drawing, processing |
| Pillow | 11.1.0 | Image format support |
| numpy | 2.1.1 | Numerical array operations |
| matplotlib | 3.10.1 | Image display & plotting |
| seaborn | 0.13.2 | Training metrics visualization |
| pandas | 2.2.3 | Data analysis for training logs |
| PyYAML | 6.0.2 | Configuration file parsing |
| scipy | 1.15.2 | Scientific computing utilities |
| pyserial | 3.5 | Serial communication with ESP32 |
| tqdm | 4.67.1 | Progress bars during training |
| streamlit | 1.55.0 | Web application interface |

## Verify Installation

After installing, verify everything works:

```bash
python -c "from ultralytics import YOLO; import cv2; import yaml; import serial; import streamlit; print('All imports OK')"
```

## ESP32 / Arduino IDE Setup

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Add ESP32 board support:
   - Go to **File → Preferences**
   - Add this URL to Board Manager URLs:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
3. Install **ESP32** from **Tools → Board → Boards Manager**
4. Select your ESP32 board under **Tools → Board**
5. Open `hardware/control_esp32.ino` and upload
