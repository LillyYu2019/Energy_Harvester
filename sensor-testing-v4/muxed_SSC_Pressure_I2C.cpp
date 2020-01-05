/******************************************************************************
  MS5803_I2C.cpp
  Library for identical SSC pressure sensors on a muxed I2C buses
  Rowan Walsh @ WERL, U of Toronto
  2017-Oct-02

    In this file are the functions in the muxedSSC class.

  Resources:
  This library uses the Arduino Wire.h library to complete I2C transactions.
******************************************************************************/

#include <Wire.h> // Wire library is used for the I2C bus
#include "muxed_SSC_Pressure_I2C.h"

muxedSSC::muxedSSC(byte muxAddress, byte sensorAddress, float pressureMax, float pressureMin, muxedSSC_transfer transFunc)
// Extended library type muxedSSC
{
	_muxAddress = muxAddress;
	_sensorAddress = sensorAddress;

	_pressureMax = pressureMax;
	_pressureMin = pressureMin;
	_transFunc = transFunc;

	staleData = false;

	Wire.begin();
}

void muxedSSC::reset(void)
// Resets device
{
	Wire.beginTransmission(_muxAddress);
	Wire.write(CMD_RESET_MUX);
	Wire.endTransmission();

	staleData = false;
}

uint16_t muxedSSC::getPressureData(muxedSSC_channel channel)
// Returns raw pressure data from specified channel
{
	selectSensor(channel);
	return readSensor();
}

float muxedSSC::convertPressureData(uint16_t pressureData)
// Converts raw pressure data to a floating point in whatever units the sensor is using
{
	float result;

	switch (_transFunc) {
		case TRANSFER_10_90:
			pressureData = pressureData - 0x0666;
			result = (float)pressureData / ((float)(0x3999 - 0x0666));
			break;
		case TRANSFER_05_95:
			pressureData = pressureData - 0x0333;
			result = (float)pressureData / ((float)(0x3CCC - 0x0333));
			break;
		case TRANSFER_05_85:
			pressureData = pressureData - 0x0333;
			result = (float)pressureData / ((float)(0x3666 - 0x0333));
			break;
		case TRANSFER_04_94:
			pressureData = pressureData - 0x028F;
			result = (float)pressureData / ((float)(0x3C28 - 0x028F));			
			break;
	}

	result = (result * (_pressureMax - _pressureMin)) + _pressureMin;
	return result;
}

void muxedSSC::selectSensor(muxedSSC_channel channel)
// Switches the mux channel to a sensor
{
	uint8_t command;

	switch (channel) {
		case CHANNEL_0:
			command = 0x01;
			break;
		case CHANNEL_1:
			command = 0x02;
			break;
		case CHANNEL_2:
			command = 0x04;
			break;
		case CHANNEL_3:
			command = 0x08;			
			break;
	}

	Wire.beginTransmission(_muxAddress);
	Wire.write(command);
	Wire.endTransmission();
}

uint16_t muxedSSC::readSensor(void)
// Reads pressure data from current channel, returns binary data, sets staleData
{
	uint8_t highByte, lowByte;
	uint8_t status;
	uint16_t result;

	Wire.requestFrom(_sensorAddress, (uint8_t)2);
	while (Wire.available())
	{
		highByte = Wire.read();
		lowByte = Wire.read();
	}

	staleData = highByte & B10000000;

	result = word((highByte & 0x3F), lowByte);
	return result;
}
