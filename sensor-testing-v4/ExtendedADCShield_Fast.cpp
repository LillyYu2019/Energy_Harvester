/******************************************************************************
	ExtendedADCShield_Fast.cpp
	Library for using Mayhew Labs' Extended ADC Shield with fast sampling
	Rowan Walsh @ WERL, U of Toronto
	2017-Oct-23

	Based on ExtendedADCShield.cpp by Mark Mayhew (June 8,2015).

	In this file are the functions in the ExtendedADCShield_Fast class

	Resources:
	This library uses <SPI.h>

******************************************************************************/

#include "Arduino.h"
#include "ExtendedADCShield_Fast.h"
#include <SPI.h>

ExtendedADCShield_Fast::ExtendedADCShield_Fast(bit_amount number_bits)
{
	_number_bits = number_bits;

	pinMode(BUSY, INPUT);
	pinMode(CONVST, OUTPUT);
	pinMode(RD, OUTPUT);

	SPI.begin();

	SPI.setClockDivider(SPI_CLOCK_DIV2); // Set for 8Mhz SPI (can be faster if using Due)
	SPI.setBitOrder(MSBFIRST);
	SPI.setDataMode(SPI_MODE0);

	digitalWrite(CONVST, LOW);
	digitalWrite(RD, LOW); // Different than ExtendedADCShield
}

void ExtendedADCShield_Fast::setConfig(uint8_t channel, uint8_t sgl_diff, uint8_t uni_bipolar, uint8_t range)
{
	_uni_bipolar = uni_bipolar;
	_range = range;

	_command = buildCommand(channel, sgl_diff, uni_bipolar, range);
}

uint16_t ExtendedADCShield_Fast::getData(void)
{
	uint8_t highByte, lowByte;

	highByte = SPI.transfer(_command);
	lowByte = SPI.transfer(B00000000); // Filler

	delayMicroseconds(2); // T_Acq time requirement

	//Trigger a conversion with a fast pulse
	noInterrupts();
	PORTB |= 0x01;
	PORTB &= ~0x01;
	interrupts();

	//Wait for conversion to be finished
	delayMicroseconds(4);

	return word(highByte, lowByte);
}

float ExtendedADCShield_Fast::convertData(uint16_t data)
{
	float voltage = 0;
	float sign = 1;

	if (_uni_bipolar == BIPOLAR) // bipolar range
	{
		if ((data & 0x8000) == 0x8000) // data is negative
		{
			data = (data ^ 0xFFFF) + (1 << (16 - _number_bits)); // Convert data from 2's-complement to binary
			sign = -1;
		}
		data = data >> (16 - _number_bits);
		voltage = sign * (float)data;
		voltage = voltage / (pow(2, _number_bits - 1) - 1);
	}
	else // unipolar range
	{
		data = data >> (16 - _number_bits);
		voltage = (float)data;
		voltage = voltage / (pow(2, _number_bits) - 1);

	}

	switch (_range)
	{
		case RANGE5V:
			voltage = voltage * 5;
			break;
		case RANGE10V:
			voltage = voltage * 10;
			break;
		default:
			break;
	}

	return voltage;
}

uint8_t ExtendedADCShield_Fast::buildCommand(byte channel, byte sgl_diff, byte uni_bipolar, byte range)
{
	uint8_t command = 0;

	switch (channel)
	{
		case 0:
			command = command | B00000000;
			break;
		case 2:
			command = command | B00010000;
			break;
		case 4:
			command = command | B00100000;
			break;
		case 6:
			command = command | B00110000;
			break;
		case 1:
			command = command | B01000000;
			break;
		case 3:
			command = command | B01010000;
			break;
		case 5:
			command = command | B01100000;
			break;
		case 7:
			command = command | B01110000;
			break;
		default:
			break;
	}

	if (sgl_diff == SINGLE_ENDED)
	{
		command = command | B10000000;
	}

	if (uni_bipolar == UNIPOLAR)
	{
		command = command | B00001000;
	}

	if (range == RANGE10V)
	{
		command = command | B00000100;
	}

	return command;
}