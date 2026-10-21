
import serial
import json
import time

# ESP32 USB Serial configuration
PORT = "COM8"
BAUD_RATE = 115200

def main():
    print(f"Connecting to ESP32 on {PORT}...")

    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=1) as esp32:
            time.sleep(2)  # Wait for ESP32 to initialize
            esp32.reset_input_buffer()

            print("Connected! Reading sensor data...")
            print("Press Ctrl+C to stop.\n")

            while True:
                line = esp32.readline().decode(
                    "utf-8", errors="ignore"
                ).strip()

                if not line:
                    continue

                # Ignore startup/debug messages
                if not line.startswith("{"):
                    print(line)
                    continue

                try:
                    data = json.loads(line)
                    print(data)

                except json.JSONDecodeError:
                    print("Invalid JSON:", line)

    except serial.SerialException as error:
        print("Serial connection error:", error)
        print("Check COM8 and close Arduino Serial Monitor.")

    except KeyboardInterrupt:
        print("\nStopped reading sensor data.")


if __name__ == "__main__":
    main()
