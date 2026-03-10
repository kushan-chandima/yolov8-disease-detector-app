# Hardware Components

This directory contains the ESP32 firmware and hardware documentation for the chili disease sprayer system.

## 🧩 Components

| Component | Model / Spec | Description |
|-----------|-------------|-------------|
| **Microcontroller** | ESP32 | Main controller handling serial commands, motor control, and pump relay |
| **Stepper Motor** | 28BYJ-48 + ULN2003 Driver | Rotates the spray nozzle to the target area (±90° or ±135°) |
| **Spray Pump** | 12V Diaphragm Pump | Delivers targeted pesticide spray to diseased leaves |
| **Power Supply** | 12V 5A SMPS | Powers the pump and control electronics |
| **Voltage Regulator** | XL4015 Buck Converter | Steps down 12V to stable operating voltage for ESP32 and logic |
| **Rain Sensor** | HW-028 | Detects rainfall to auto-stop spraying and protect the system |

## 🔌 Pin Wiring

| ESP32 Pin | Connected To |
|-----------|-------------|
| D13 | Stepper Motor IN1 |
| D12 | Stepper Motor IN2 |
| D14 | Stepper Motor IN3 |
| D27 | Stepper Motor IN4 |
| D2 | Pump Relay |
| D35 | Rain Sensor (analog input) |

## 🔄 System Flow

```
Python Detection Script
        │
        ▼
  Serial Command ('1'-'4')
        │
        ▼
    ESP32 Controller
        │
   ┌────┴────┐
   ▼         ▼
Stepper    Pump Relay
Motor      (ON/OFF)
   │
   ▼
Spray Nozzle → Target Area
```

## 📎 Notes

- Upload `control_esp32.ino` using Arduino IDE with ESP32 board package installed.
- Set Serial Monitor baud rate to **115200**.
- The rain sensor automatically stops spraying and returns to the home position when moisture is detected.
