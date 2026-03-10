/*
 * ESP32 Chili Disease Sprayer Controller
 * ========================================
 * Controls a stepper motor (28BYJ-48) for directional aiming
 * and a diaphragm pump for targeted pesticide spraying.
 * 
 * Serial Commands:
 *   '0' -> Reset to home position, pump OFF
 *   '1' -> Rotate +90°,  pump ON
 *   '2' -> Rotate -90°,  pump ON
 *   '3' -> Rotate +135°, pump ON
 *   '4' -> Rotate -135°, pump ON
 * 
 * Rain sensor (HW-028) auto-stops spraying and resets on rain detection.
 * 
 * Hardware:
 *   - ESP32 microcontroller
 *   - 28BYJ-48 Stepper Motor + ULN2003 Driver
 *   - 12V Diaphragm Pump via relay
 *   - HW-028 Rain Sensor
 */

#include <Stepper.h>

// --- Pin Definitions ---
#define IN1 13                  // Stepper motor coil 1 (D13)
#define IN2 12                  // Stepper motor coil 2 (D12)
#define IN3 14                  // Stepper motor coil 3 (D14)
#define IN4 27                  // Stepper motor coil 4 (D27)

#define PUMP_PIN 2              // Pump relay control (D2)
#define RAIN_SENSOR_PIN 35      // Rain sensor input (D35)

// --- Motor Configuration ---
#define STEPS_PER_REV 2048      // 28BYJ-48 steps per revolution (64:1 gear ratio)

// Create stepper motor instance (correct coil sequence for 28BYJ-48)
Stepper stepperMotor(STEPS_PER_REV, IN1, IN3, IN2, IN4);

// --- State Variables ---
int  currentPosition = 0;       // Current step position
bool pumpActive      = false;   // Pump/spray status

// =====================================================================
// SETUP
// =====================================================================
void setup() {
    Serial.begin(115200);
    stepperMotor.setSpeed(10);  // RPM

    pinMode(PUMP_PIN, OUTPUT);
    pinMode(RAIN_SENSOR_PIN, INPUT);

    digitalWrite(PUMP_PIN, LOW);  // Pump off initially
    Serial.println("ESP32 Sprayer Controller ready.");
}

// =====================================================================
// HELPER: Move stepper to target position (in steps)
// =====================================================================
void moveToPosition(int targetSteps) {
    int moveSteps = targetSteps - currentPosition;
    stepperMotor.step(moveSteps);
    currentPosition = targetSteps;

    Serial.print("Moved to ");
    Serial.print(currentPosition * 360.0 / STEPS_PER_REV);
    Serial.println(" degrees.");
}

// =====================================================================
// HELPER: Reset to home position and stop pump
// =====================================================================
void resetToHome(const char* reason) {
    pumpActive = false;
    digitalWrite(PUMP_PIN, LOW);
    delay(1000);

    Serial.print(reason);
    Serial.println(" Returning to home position...");

    stepperMotor.step(-currentPosition);
    currentPosition = 0;

    Serial.println("Home position set.");
}

// =====================================================================
// MAIN LOOP
// =====================================================================
void loop() {
    // --- Process serial commands from Python ---
    if (Serial.available()) {
        char command = Serial.read();

        switch (command) {
            case '0':
                resetToHome("Manual reset.");
                return;

            case '1':
                moveToPosition((90 * STEPS_PER_REV) / 360);
                digitalWrite(PUMP_PIN, HIGH);
                pumpActive = true;
                Serial.println("Area 1: +90 deg, pump ON.");
                break;

            case '2':
                moveToPosition((-90 * STEPS_PER_REV) / 360);
                digitalWrite(PUMP_PIN, HIGH);
                pumpActive = true;
                Serial.println("Area 2: -90 deg, pump ON.");
                break;

            case '3':
                moveToPosition((135 * STEPS_PER_REV) / 360);
                digitalWrite(PUMP_PIN, HIGH);
                pumpActive = true;
                Serial.println("Area 3: +135 deg, pump ON.");
                break;

            case '4':
                moveToPosition((-135 * STEPS_PER_REV) / 360);
                digitalWrite(PUMP_PIN, HIGH);
                pumpActive = true;
                Serial.println("Area 4: -135 deg, pump ON.");
                break;

            default:
                Serial.println("Invalid command. Send 0-4.");
                return;
        }
    }

    // --- Rain sensor auto-stop ---
    if (pumpActive && digitalRead(RAIN_SENSOR_PIN) == LOW) {
        resetToHome("Rain detected! Stopping spray.");
    }
}
