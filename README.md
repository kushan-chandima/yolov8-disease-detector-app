# 🌶️ YOLOv8 Chili Disease Detection & Control System

A smart chili plant disease detection and targeted pesticide spraying system using **YOLOv8 object detection** and an **ESP32-controlled mini sprayer** to reduce chemical usage and improve crop health.

## 📋 Overview

This project automates the identification and control of **Leaf Curl disease** in chili plants:

1. A top-view camera captures an image of the chili plant
2. The **YOLOv8 model** detects diseased leaves in the image
3. The system determines which quadrant contains the disease
4. An **ESP32 microcontroller** rotates a stepper motor to aim the spray nozzle
5. A **diaphragm pump** delivers pesticide only to the affected area

> This targeted approach reduces chemical waste and improves efficiency in small-scale crop fields.

## 🏗️ System Architecture

```
Camera → YOLOv8 Detection → Quadrant Mapping → ESP32 (Serial) → Stepper Motor + Pump → Targeted Spray
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 85.6% |
| Precision | 87.5% |
| mAP50 | 75.2% |
| Dataset | 440 real field images |
| Classes | 1 (Leaf Curl Disease) |

## 📁 Project Structure

```
├── config/
│   └── config.yaml              # Centralized configuration
├── hardware/
│   ├── control_esp32.ino        # ESP32 motor & pump firmware
│   └── README.md                # Hardware components & wiring
├── models/                      # Trained model weights (.pt files)
├── scripts/
│   ├── detect.py                # Disease detection module
│   └── train.py                 # Model training script
├── data/                        # Dataset images
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
├── setup_env.md                 # Environment setup guide
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

See [setup_env.md](setup_env.md) for detailed setup instructions.

### 2. Train the Model

Place your dataset in `data/` with the following structure:
```
data/
├── images/    # Training images
├── valid/     # Validation images
└── labels/    # YOLO format annotations
```

```bash
python scripts/train.py
```

### 4. Run Global Detection (Web UI)

For a user-friendly interface with image upload and visualization:

```bash
streamlit run app.py
```

### 5. CLI Detection (Optional)

Place your trained model weights (`best.pt`) in the `models/` directory, then:

```bash
python scripts/detect.py
```

The script will prompt for an image path, run detection, display results, and send spray commands to the ESP32.

### 4. ESP32 Setup

1. Open `hardware/control_esp32.ino` in **Arduino IDE**
2. Install the ESP32 board package
3. Upload the firmware to your ESP32
4. Connect via USB (default: COM3, 115200 baud)

See [hardware/README.md](hardware/README.md) for wiring and component details.

## ⚙️ Configuration

All settings are centralized in [`config/config.yaml`](config/config.yaml):

- **Model:** weights path, confidence threshold, image size
- **Detection:** bounding box colors, quadrant boundaries
- **Serial:** COM port, baud rate, timeouts
- **Training:** epochs, image size, base model, output directories
- **Dataset:** paths, class names, number of classes

## 🛠️ Software & Tools

| Tool | Purpose |
|------|---------|
| Python 3.10.8 | Core programming language |
| Streamlit | Web application interface |
| Ultralytics YOLOv8 | Object detection framework |
| OpenCV | Image processing |
| PyTorch | Deep learning backend |
| Roboflow | Dataset annotation |
| Arduino IDE | ESP32 firmware upload |

## 📹 Demo

🎥 Watch the prototype demonstration: [YouTube Demo](https://youtu.be/Fit3X5Yvql0)

## 📄 License

This project is licensed under the MIT License.
