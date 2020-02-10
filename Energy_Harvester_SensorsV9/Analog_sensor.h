/******************************************************************************
	Analog_sensor.h
	Library for reading averaged analog signals
	Lilly Yu @ WERL, U of Toronto (adapted from Rowan Walsh's version)
	2020-02-07


******************************************************************************/

#ifndef Analog_sensor_h
#define Analog_sensor_h

#include "Arduino.h"

class Analog_sensor
{
public:
	//constructors
	Analog_sensor();
	Analog_sensor(int pin, int analog_target_samples, float analog_range[]);

	// functions
	void read();
	float get_readings();
	float get_sample_time();

private:
	//private variables
	int pin_;
	int target_sample_count_;
	float *analog_range_;
	float read_count_;
	float voltage_;
	float reading_;
	float start_time_;
	float sample_time_;

	//private functions
	void clear();
	void map();
};

#endif // Analog_sensor_h
