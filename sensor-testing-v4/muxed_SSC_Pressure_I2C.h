/******************************************************************************
  MS5803_I2C.h
  Library for identical SSC pressure sensors on a muxed I2C buses
  Rowan Walsh @ WERL, U of Toronto
  2017-Oct-02

    In this file are the function definitions in the muxedSSC class.

  Resources:
  This library uses the Arduino Wire.h library to complete I2C transactions.
******************************************************************************/

#ifndef muxedSSC_I2C_h
#define muxedSSC_I2C_h

#include <Arduino.h>

// Define enumerated types
enum muxedSSC_transfer
{
	TRANSFER_10_90,	// A type, 10%-90% of full-range
	TRANSFER_05_95,	// B type, 5%-95% of full-range
	TRANSFER_05_85,	// C type, 5%-85% of full-range
	TRANSFER_04_94,	// F type, 4%-94% of full-range
};

enum muxedSSC_channel
{
	CHANNEL_0, 
	CHANNEL_1, 
	CHANNEL_2, 
	CHANNEL_3, 
};

// Define commands
#define CMD_RESET_MUX 0x00

class muxedSSC
{
	public:
		bool staleData;

		muxedSSC(uint8_t muxAddress, uint8_t sensorAddress, float pressureMax=30.0, float pressureMin=0.0, muxedSSC_transfer transFunc=TRANSFER_10_90);
		void reset(void);	// Reset device
		uint16_t getPressureData(muxedSSC_channel channel);	// Returns raw pressure data from specified channel
		float convertPressureData(uint16_t pressureData);	// Converts raw pressure data to a floating point

	private:
		uint8_t _muxAddress, _sensorAddress;
		uint16_t _pressureReading;
		float _pressureMax, _pressureMin;
		muxedSSC_transfer _transFunc;

		void selectSensor(muxedSSC_channel channel);	// Switches the mux channel to a sensor
		uint16_t readSensor(void); // Reads pressure data from current channel, returns binary data, sets _staleData
};

#endif
