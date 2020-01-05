/******************************************************************************
	ExtendedADCShield_Fast.h
	Library for using Mayhew Labs' Extended ADC Shield with fast sampling of a single channel
	Rowan Walsh @ WERL, U of Toronto
	2017-Oct-23

	Based on ExtendedADCShield.h by Mark Mayhew (June 8,2015).

	In this file are the function definitions for the ExtendedADCShield_Fast class

	Resources:
	This library uses <SPI.h>

******************************************************************************/

#ifndef ExtendedADCShield_Fast_h
#define ExtendedADCShield_Fast_h

#include "Arduino.h"

#define SINGLE_ENDED 0
#define DIFFERENTIAL 1
#define UNIPOLAR 0
#define BIPOLAR 1
#define RANGE5V 0
#define RANGE10V 1

#define BUSY 9
#define CONVST 8
#define RD 10

enum bit_amount
{
	BITS_16 = 16,
	BITS_14 = 14,
	BITS_12 = 12,
};

class ExtendedADCShield_Fast
{
	public:
		ExtendedADCShield_Fast(bit_amount number_bits);
		void setConfig(uint8_t channel, uint8_t sgl_diff, uint8_t uni_bipolar, uint8_t range);
		uint16_t getData(void);
		float convertData(uint16_t data);

	private:
		uint8_t _number_bits, _uni_bipolar, _range;
		uint8_t _command;
		uint8_t buildCommand(byte channel, byte sgl_diff, byte uni_bipolar, byte range);
};

#endif // ExtendedADCShield_Fast_h