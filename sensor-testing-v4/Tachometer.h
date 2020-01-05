/******************************************************************************
	Tachometer.h
	Library for reading angular speed from a tachometer (square-wave) signal
	Rowan Walsh @ WERL, U of Toronto
	2017-Oct-23

	The tachometer signal is assumed to be composed of square pulses

	In this file are the function definitions in the Tachometer class

	Resources:
	This library uses interupts

******************************************************************************/

#ifndef Tachometer_h
#define Tachometer_h

#include "Arduino.h"

// Define units for conversions.
enum angular_speed_units
{
	RAD_PER_SECOND,
	ROT_PER_MINUTE,
};

// Define possible digital pins for sensor
// (only interupt-capable pins)
enum tachometer_pin
{
	PIN_D2 = 2,
	PIN_D3 = 3,
};

static volatile bool tachometer_risingEdgeWait;

class Tachometer
{
public:
	Tachometer(tachometer_pin pin, uint16_t tachometerPolePairs=1, uint16_t maxSampleTimeMillis=100, uint16_t targetSampleCount=1);

	void setMaxSampleTime(uint16_t maxSampleTimeMillis);
	void setTargetSampleCount(uint16_t targetSampleCount);

	// Return calculated rad/s from tachometer
	float getAngularSpeed(angular_speed_units units);

private:
	uint16_t _tachometerPolePairs;
	uint16_t _maxSampleTimeMillis;
	uint16_t _targetSampleCount;
	tachometer_pin _pin;

	// ISR for counting tachometer rising edges
	static void incrementRisingEdge();

};

#endif // Tachometer_h
