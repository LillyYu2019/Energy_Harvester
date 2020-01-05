/******************************************************************************
	Tachometer.cpp
	Library for reading angular speed from a tachometer (square-wave) signal
	Rowan Walsh @ WERL, U of Toronto
	2017-Oct-23

	The tachometer signal is assumed to be composed of square pulses

	In this file are the functions in the Tachometer class

	Resources:
	This library uses interupts

******************************************************************************/

#include "Arduino.h"
#include "Tachometer.h"

Tachometer::Tachometer(tachometer_pin pin, uint16_t tachometerPolePairs, uint16_t maxSampleTimeMillis, uint16_t targetSampleCount)
{
	_pin = pin;
	_tachometerPolePairs = tachometerPolePairs;
	_maxSampleTimeMillis = maxSampleTimeMillis;
	_targetSampleCount = targetSampleCount;
}

// Simple helper function to change the sampling time
void Tachometer::setMaxSampleTime(uint16_t maxSampleTimeMillis)
{
	_maxSampleTimeMillis = maxSampleTimeMillis;
}

// Simple helper function to change the sampling time
void Tachometer::setTargetSampleCount(uint16_t targetSampleCount)
{
	_targetSampleCount = targetSampleCount;
}

float Tachometer::getAngularSpeed(angular_speed_units units)
{
	bool timeOutFlag = false;
	uint32_t startMillis, startMicros, durationMicros;
	float rotationsPerSecond, angular_speed_reported;

	tachometer_risingEdgeWait = true;
	startMillis = millis();
	switch ( _pin ) // Clear interrupt flag so previous rotations do not get counted
	{
		case PIN_D2 : EIFR |= (1 << INTF0); break;
		case PIN_D3 : EIFR |= (1 << INTF1); break;
	}
	attachInterrupt(digitalPinToInterrupt(_pin), incrementRisingEdge, RISING);

	// Wait for a rising edge to occur
	while (tachometer_risingEdgeWait) // Wait until a rising edge occurs
	{
		if (millis()-startMillis < _maxSampleTimeMillis)
		{
			timeOutFlag = true;
			break;
		}
	}
	startMicros = micros(); // Set start time

	// Then count the target number of rising edges
	for (uint8_t i=0; i<_targetSampleCount; i++)
	{
		tachometer_risingEdgeWait = true;
		while (tachometer_risingEdgeWait) // Wait until another rising edge occurs
		{
			if (millis()-startMillis < _maxSampleTimeMillis)
			{
				timeOutFlag = true;
				break;
				break;
			}
		}
	}
	durationMicros = micros() - startMicros;

	detachInterrupt(digitalPinToInterrupt(_pin));

	if (timeOutFlag)
	{
		rotationsPerSecond = 0;
	}
	else
	{
		rotationsPerSecond = 1000000.0 *_targetSampleCount / (float(durationMicros * _tachometerPolePairs));
	}

	if (units == ROT_PER_MINUTE)
	{
		angular_speed_reported = 60.0 * rotationsPerSecond;
	}
	else // units == RAD_PER_SECOND
	{
		angular_speed_reported = 2.0 * 3.141592653589 * rotationsPerSecond;
	}

	return angular_speed_reported;
}

void Tachometer::incrementRisingEdge()
{
	tachometer_risingEdgeWait = false;
}
