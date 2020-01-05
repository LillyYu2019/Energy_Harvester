/******************************************************************************
	sensor-testing-v4.ino
	Program to collect a range of data from the in-situ Dive Device sensors
	Rowan Walsh @ WERL, U of Toronto
	2017-Nov-08

	Required connections:
		- Serial connection to a computer/serial monitor for data collection, use a serial baud-rate of 115200
				USB connection supplies Arduino with power
		- Connection to pressure sensor enclosure

******************************************************************************/

#include <Arduino.h>
#include "muxed_SSC_Pressure_I2C.h"
#include "ExtendedADCShield_Fast.h"
#include "Tachometer.h"

#define SERIAL_BAUDRATE 115200				// Serial output baud-rate

#define MUX_I2C_ADDRESS 0x70				// Mux address (0x70 to 0x77)
#define PRES_ADDRESS 0x28					// Pressure sensor (duplicate) address
#define PRES_MAX_READING 30					// Pressure sensor pressure range maximum
#define PRES_MIN_READING 0					// Pressure sensor pressure range minimum

#define TACHOMETER_TIMEOUT_MILLIS 500		// Timeout duration before defaulting to zero angular velocity
#define TACHOMETER_EDGES_PER_ROTATION 7		// Number of rising edges per single shaft revolution
#define TACHOMETER_EDGES_TO_COUNT 35		// Number of rising edges to average duration over

#define EXT_ADC_PIN 0

#define LOOP_DELAY_ALL_PRES_US 0			// Delay between readings from 4 pressure sensors (max 16383)
#define LOOP_DELAY_ONE_PRES_US 0			// Delay between readings from 1 pressure sensor (max 16383)
#define LOOP_DELAY_TORQUE_US 0				// Delay between readings from torque sensor (max 16383)
#define LOOP_DELAY_ALL_US 0           // Delay between readings from torque and all 4 pres sensors (max 16383)

#define OUTPUT_TORQUE_DECIMALS 4			// Decimal places to use when writing torque to serial
#define OUTPUT_PRES_DECIMALS 4				// Decimal places to use when writing pressure to serial
#define OUTPUT_SPEED_DECIMALS 1				// Decimal places to use when writing angular speed to serial
//#define OUTPUT_READING_AMOUNT 150			// Amount of readings to do in 4-pres fast-sample mode
#define OUTPUT_READING_AMOUNT 600     // Amount of data readings to store (maximum)
#define PRES_SENSOR_N 4               // Number of pressure sensors

enum serial_commands
{
	SERIAL_COMMAND_0 = 48,
	SERIAL_COMMAND_1 = 49,
	SERIAL_COMMAND_2 = 50,
	SERIAL_COMMAND_3 = 51,
	SERIAL_COMMAND_4 = 52,
	SERIAL_COMMAND_5 = 53,
};

// Define sensor objects
Tachometer tachometerSensor(PIN_D2, TACHOMETER_EDGES_PER_ROTATION, TACHOMETER_TIMEOUT_MILLIS, TACHOMETER_EDGES_TO_COUNT);
muxedSSC pressureSensors(MUX_I2C_ADDRESS, PRES_ADDRESS, PRES_MAX_READING, PRES_MIN_READING, TRANSFER_10_90);
ExtendedADCShield_Fast torqueSensor(BITS_16);

// Define global loop variables
uint16_t data[OUTPUT_READING_AMOUNT];
float angularSpeedReading;
uint32_t readStartMicros, readDurationMicros;
uint8_t serialCommand = 0;

void setup() {
	// Initialize serial output
	Serial.begin(SERIAL_BAUDRATE);

	// Initialize library objects
	pressureSensors.reset();
	torqueSensor.setConfig(EXT_ADC_PIN, DIFFERENTIAL, BIPOLAR, RANGE5V);
}


void loop() {
	// Check if a command is available in the incoming serial data
	if (Serial.available() > 0) {
		// Read incoming byte
		serialCommand = Serial.read();
	
		switch (serialCommand) {
			case SERIAL_COMMAND_0: // Print list of commands
				Serial.println(F("0\tList of commands"));
				Serial.println(F("1\tSample all sensors once"));
				Serial.println(F("2\tSample 4 pressure sensors quickly"));
				Serial.println(F("3\tSample single pressure sensor (0) quickly"));
        Serial.println(F("4\tSample torque sensor quickly"));
        Serial.println(F("5\tSample torque and 4 pressure sensors quickly"));
				break;

			case SERIAL_COMMAND_1: // Sample all sensors once
				// Read in values from the sensors
				data[0] = torqueSensor.getData();
				angularSpeedReading = tachometerSensor.getAngularSpeed(RAD_PER_SECOND);
				data[1] = pressureSensors.getPressureData(CHANNEL_0);
				data[2] = pressureSensors.getPressureData(CHANNEL_1);
				data[3] = pressureSensors.getPressureData(CHANNEL_2);
				data[4] = pressureSensors.getPressureData(CHANNEL_3);

				// Report values via serial
				Serial.println(F("T,\t,w,\tP0,\tP1,\tP2,\tP3"));
				Serial.println(F("(N),\t(ra/s),\t(psi),\t(psi),\t(psi),\t(psi)"));

				Serial.print(torqueSensor.convertData(data[0]), OUTPUT_TORQUE_DECIMALS);
				Serial.print(F(",\t"));
				Serial.print(angularSpeedReading, OUTPUT_SPEED_DECIMALS);
				Serial.print(F(",\t"));
				for (int i=1; i<1+PRES_SENSOR_N; i++) {
					// Convert pressure data to floating point (psi)
					Serial.print(pressureSensors.convertPressureData(data[i]), OUTPUT_PRES_DECIMALS);
					Serial.print(F(",\t"));
				}
				Serial.println();
				break;

			case SERIAL_COMMAND_2: // Sample all 4 pressure sensors quickly
				readStartMicros = micros();
				for (int j=0; j<OUTPUT_READING_AMOUNT/PRES_SENSOR_N; j++) {
					data[0 + PRES_SENSOR_N*j] = pressureSensors.getPressureData(CHANNEL_0);
					data[1 + PRES_SENSOR_N*j] = pressureSensors.getPressureData(CHANNEL_1);
					data[2 + PRES_SENSOR_N*j] = pressureSensors.getPressureData(CHANNEL_2);
					data[3 + PRES_SENSOR_N*j] = pressureSensors.getPressureData(CHANNEL_3);

					delayMicroseconds(LOOP_DELAY_ALL_PRES_US);
				}
				readDurationMicros = micros() - readStartMicros;

				// Report values via serial
				Serial.println(F("P0,\tP1,\tP2,\tP3"));
				Serial.println(F("(psi),\t(psi),\t(psi),\t(psi)"));

				for (int j=0; j<OUTPUT_READING_AMOUNT/PRES_SENSOR_N; j++) {
					for (int i=0; i<PRES_SENSOR_N; i++) {
						// Convert pressure data to floating point (psi)
						Serial.print(pressureSensors.convertPressureData(data[i + PRES_SENSOR_N*j]), OUTPUT_PRES_DECIMALS);
						Serial.print(F(",\t"));
					}
					Serial.println();
				}
				Serial.println();
				Serial.print(readDurationMicros);
				Serial.print(F(", "));
				Serial.print(OUTPUT_READING_AMOUNT/PRES_SENSOR_N);
				Serial.print(F(", "));
				Serial.print(tachometerSensor.getAngularSpeed(RAD_PER_SECOND), OUTPUT_SPEED_DECIMALS);
				Serial.println();
				break;

			case SERIAL_COMMAND_3: // Sample single pressure sensor quickly
				readStartMicros = micros();
				for (int j=0; j<OUTPUT_READING_AMOUNT; j++) {
					data[j] = pressureSensors.getPressureData(CHANNEL_0);

					delayMicroseconds(LOOP_DELAY_ONE_PRES_US);
				}
				readDurationMicros = micros() - readStartMicros;

				// Report values via serial
				Serial.println(F("P0"));
				Serial.println(F("(psi)"));

				for (int j=0; j<OUTPUT_READING_AMOUNT; j++) {
					// Convert pressure data to floating point (psi)
					Serial.print(pressureSensors.convertPressureData(data[j]), OUTPUT_PRES_DECIMALS);
					Serial.println();
				}
				Serial.println();
				Serial.print(readDurationMicros);
				Serial.print(F(", "));
				Serial.print(OUTPUT_READING_AMOUNT);
				Serial.print(F(", "));
				Serial.print(tachometerSensor.getAngularSpeed(RAD_PER_SECOND), OUTPUT_SPEED_DECIMALS);
				Serial.println();
				break;

			case SERIAL_COMMAND_4: // Sample torque sensor quickly
				readStartMicros = micros();
				for (int j=0; j<OUTPUT_READING_AMOUNT; j++) {
					data[j] = torqueSensor.getData();

					delayMicroseconds(LOOP_DELAY_TORQUE_US);
				}
				readDurationMicros = micros() - readStartMicros;

				// Report values via serial
				Serial.println(F("Torque"));
				Serial.println(F("(N)"));

				for (int j=0; j<OUTPUT_READING_AMOUNT; j++) {
					// Convert pressure data to floating point (psi)
					Serial.print(torqueSensor.convertData(data[j]), OUTPUT_TORQUE_DECIMALS);
					Serial.println();
				}
				Serial.println();
				Serial.print(readDurationMicros);
				Serial.print(F(", "));
				Serial.print(OUTPUT_READING_AMOUNT);
				Serial.print(F(", "));
				Serial.print(tachometerSensor.getAngularSpeed(RAD_PER_SECOND), OUTPUT_SPEED_DECIMALS);
				Serial.println();
				break;
        
    case SERIAL_COMMAND_5: // Sample torque and 4 pressure sensors quickly
        readStartMicros = micros();
        for (int j=0; j<(OUTPUT_READING_AMOUNT/(PRES_SENSOR_N+1)); j++) {
          data[0 + (PRES_SENSOR_N+1)*j] = torqueSensor.getData();
          data[1 + (PRES_SENSOR_N+1)*j] = pressureSensors.getPressureData(CHANNEL_0);
          data[2 + (PRES_SENSOR_N+1)*j] = pressureSensors.getPressureData(CHANNEL_1);
          data[3 + (PRES_SENSOR_N+1)*j] = pressureSensors.getPressureData(CHANNEL_2);
          data[4 + (PRES_SENSOR_N+1)*j] = pressureSensors.getPressureData(CHANNEL_3);

          delayMicroseconds(LOOP_DELAY_ALL_US);
        }
        readDurationMicros = micros() - readStartMicros;

        // Report values via serial
        Serial.println(F("T,\tP0,\tP1,\tP2,\tP3"));
        Serial.println(F("(N),\t(psi),\t(psi),\t(psi),\t(psi)"));

        for (int j=0; j<(OUTPUT_READING_AMOUNT/(PRES_SENSOR_N+1)); j++) {
          Serial.print(torqueSensor.convertData(data[(PRES_SENSOR_N+1)*j]), OUTPUT_TORQUE_DECIMALS);
          Serial.print(F(",\t"));
          for (int i=0; i<PRES_SENSOR_N; i++) {
            // Convert pressure data to floating point (psi)
            Serial.print(pressureSensors.convertPressureData(data[1 + i + (PRES_SENSOR_N+1)*j]), OUTPUT_PRES_DECIMALS);
            Serial.print(F(",\t"));
          }
          Serial.println();
        }
        Serial.println();
        Serial.print(readDurationMicros);
        Serial.print(F(", "));
        Serial.print(OUTPUT_READING_AMOUNT/(PRES_SENSOR_N+1));
        Serial.print(F(", "));
        Serial.print(tachometerSensor.getAngularSpeed(RAD_PER_SECOND), OUTPUT_SPEED_DECIMALS);
        Serial.println();
        break;
        
			default: // Invalid command
				Serial.print(F("\tCommand not recognized."));
				Serial.println();
				break;
		}

		Serial.println();
		Serial.println(F("Enter command, '0' for list of commands."));
		Serial.flush();
		serialCommand = -1;
	}
}

