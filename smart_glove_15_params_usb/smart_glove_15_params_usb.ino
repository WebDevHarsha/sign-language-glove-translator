#include <Wire.h>
#include <math.h>

const int PIN_THUMB = 36, PIN_INDEX = 39, PIN_MIDDLE = 34;
const int PIN_RING = 35, PIN_LITTLE = 32;
const int SDA_PIN = 21, SCL_PIN = 22;
const uint8_t MPU_ADDR = 0x68;
const unsigned long SAMPLE_INTERVAL_MS = 20;
unsigned long lastSample = 0;

int16_t accelX = 0, accelY = 0, accelZ = 0;
int16_t gyroX = 0, gyroY = 0, gyroZ = 0;
float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
unsigned long previousIMUTime = 0;

void writeMPU(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readMPU(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)1) != 1) return 0xFF;
  return Wire.read();
}

bool readMPUBytes(uint8_t reg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, length) != length) return false;
  for (uint8_t i = 0; i < length; i++) buffer[i] = Wire.read();
  return true;
}

bool initializeMPU() {
  writeMPU(0x6B, 0x00);  // Wake sensor
  delay(100);
  writeMPU(0x19, 0x04);
  writeMPU(0x1A, 0x03);
  writeMPU(0x1B, 0x00);  // Gyro +/-250 deg/s
  writeMPU(0x1C, 0x00);  // Accelerometer +/-2g

  uint8_t id = readMPU(0x75);
  Serial.print("# WHO_AM_I = 0x");
  Serial.println(id, HEX);
  if (id != 0x68) {
    Serial.println("# WARNING: unexpected sensor ID; trying to continue");
  }

  uint8_t test[14];
  if (!readMPUBytes(0x3B, test, 14)) {
    Serial.println("# ERROR: MPU data registers cannot be read");
    return false;
  }
  return true;
}

bool readMotionSensor() {
  uint8_t b[14];
  if (!readMPUBytes(0x3B, b, 14)) return false;

  accelX = (int16_t)(((uint16_t)b[0] << 8) | b[1]);
  accelY = (int16_t)(((uint16_t)b[2] << 8) | b[3]);
  accelZ = (int16_t)(((uint16_t)b[4] << 8) | b[5]);
  gyroX  = (int16_t)(((uint16_t)b[8] << 8) | b[9]);
  gyroY  = (int16_t)(((uint16_t)b[10] << 8) | b[11]);
  gyroZ  = (int16_t)(((uint16_t)b[12] << 8) | b[13]);
  return true;
}

void updateQuaternion() {
  unsigned long now = micros();
  float dt = (now - previousIMUTime) / 1000000.0f;
  previousIMUTime = now;
  if (dt <= 0.0f || dt > 0.1f) dt = 0.02f;

  float ax = accelX / 16384.0f;
  float ay = accelY / 16384.0f;
  float az = accelZ / 16384.0f;
  float gx = gyroX / 131.0f;
  float gy = gyroY / 131.0f;
  float gz = gyroZ / 131.0f;

  float accelRoll = atan2f(ay, az) * 180.0f / PI;
  float accelPitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
  const float alpha = 0.98f;

  roll = alpha * (roll + gx * dt) + (1.0f - alpha) * accelRoll;
  pitch = alpha * (pitch + gy * dt) + (1.0f - alpha) * accelPitch;
  yaw += gz * dt;

  float r = roll * PI / 360.0f;
  float p = pitch * PI / 360.0f;
  float y = yaw * PI / 360.0f;
  float cr = cosf(r), sr = sinf(r);
  float cp = cosf(p), sp = sinf(p);
  float cy = cosf(y), sy = sinf(y);

  qw = cr * cp * cy + sr * sp * sy;
  qx = sr * cp * cy - cr * sp * sy;
  qy = cr * sp * cy + sr * cp * sy;
  qz = cr * cp * sy - sr * sp * cy;
}

void sendSensorData() {
  int thumb = analogReadMilliVolts(PIN_THUMB);
  int index = analogReadMilliVolts(PIN_INDEX);
  int middle = analogReadMilliVolts(PIN_MIDDLE);
  int ring = analogReadMilliVolts(PIN_RING);
  int little = analogReadMilliVolts(PIN_LITTLE);

  if (!readMotionSensor()) {
    Serial.println("# ERROR: MPU read failed");
    return;
  }
  updateQuaternion();

  // Exactly 15 numeric parameters, one JSON object per line.
  Serial.print("{\"thumb_mV\":"); Serial.print(thumb);
  Serial.print(",\"index_mV\":"); Serial.print(index);
  Serial.print(",\"middle_mV\":"); Serial.print(middle);
  Serial.print(",\"ring_mV\":"); Serial.print(ring);
  Serial.print(",\"little_mV\":"); Serial.print(little);
  Serial.print(",\"quat_w\":"); Serial.print(qw, 6);
  Serial.print(",\"quat_x\":"); Serial.print(qx, 6);
  Serial.print(",\"quat_y\":"); Serial.print(qy, 6);
  Serial.print(",\"quat_z\":"); Serial.print(qz, 6);
  Serial.print(",\"gyro_x\":"); Serial.print(gyroX);
  Serial.print(",\"gyro_y\":"); Serial.print(gyroY);
  Serial.print(",\"gyro_z\":"); Serial.print(gyroZ);
  Serial.print(",\"accel_x\":"); Serial.print(accelX);
  Serial.print(",\"accel_y\":"); Serial.print(accelY);
  Serial.print(",\"accel_z\":"); Serial.print(accelZ);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("# SMART GLOVE USB SERIAL");
  Serial.println("# JSON records follow; use 115200 baud");

  analogReadResolution(12);
  analogSetPinAttenuation(PIN_THUMB, ADC_11db);
  analogSetPinAttenuation(PIN_INDEX, ADC_11db);
  analogSetPinAttenuation(PIN_MIDDLE, ADC_11db);
  analogSetPinAttenuation(PIN_RING, ADC_11db);
  analogSetPinAttenuation(PIN_LITTLE, ADC_11db);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  if (!initializeMPU()) {
    Serial.println("# MPU initialization failed. Check wiring.");
    while (true) delay(1000);
  }

  previousIMUTime = micros();
  Serial.println("# Starting sensor stream");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
    sendSensorData();
  }
}
