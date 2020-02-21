/******************************************************************************
	Tachometer.h
	Library for reading angular speed from a tachometer (square-wave) signal
	Lilly Yu @ WERL, U of Toronto (adapted from Rowan Walsh's version)
	2020-02-07

	The tachometer signal is assumed to be composed of square pulses

	In this file are the functions in the Tachometer class


******************************************************************************/

#ifndef Tachometer_h
#define Tachometer_h

#include "Arduino.h"

class Tachometer
{
public:
	//constructors
	Tachometer();
	Tachometer(float digital_k, int digital_target_samples, float time_out_limit);

	// functions
	void read();
	void increment_edge();
	float get_readings();
	float get_sample_time(); //in ms

private:
	//private variables
	int tachometer_edges_;
	int target_sample_count_;
	float k_;
	float time_out_limit_;
	float start_micros_;
	float duration_micros_;
	float reading_;

	//private functions
	void clear();
};

#endif // Tachometer_h
