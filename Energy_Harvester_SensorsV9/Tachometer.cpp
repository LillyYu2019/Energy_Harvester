/******************************************************************************
	Tachometer.cpp
	Library for reading angular speed from a tachometer (square-wave) signal
	Lilly Yu @ WERL, U of Toronto (adapted from Rowan Walsh's version)
	2020-02-07

	The tachometer signal is assumed to be composed of square pulses

	In this file are the functions in the Tachometer class

******************************************************************************/

#include "Arduino.h"
#include "Tachometer.h"

Tachometer::Tachometer() {}

Tachometer::Tachometer(float digitalk_, int digital_target_samples, float time_out_limit)
{
	k_ = digitalk_;
	target_sample_count_ = digital_target_samples;
	time_out_limit_ = time_out_limit;

	tachometer_edges_ = 0;

	clear();

	startMicros_ = micros();
}

float Tachometer::getreading_s()
{
	return reading_;
}

float Tachometer::get_sample_time()
{
	return durationMicros_ / 1000.0;
}

void Tachometer::clear()
{
	reading_ = 0.0;
	durationMicros_ = 0.0;
}

void Tachometer::read()
{
	if (tachometer_edges_ >= target_sample_count_)
	{
		int new_target = tachometer_edges_ + 1;
		bool time_out_flag = false;

		while (tachometer_edges_ < new_target) // Wait until an edge occurs
		{
			if ((micros() - startMicros_) / 1000.0 > time_out_limit_)
			{
				time_out_flag = true;
				break;
			}
		}

		durationMicros_ = micros() - startMicros_;

		while (tachometer_edges_ < 1) // Wait until an edge occurs
		{
			if ((micros() - startMicros_) / 1000.0 > time_out_limit_)
			{
				time_out_flag = true;
				break;
			}
		}

		startMicros_ = micros();
		tachometer_edges_ = 1;

		if (time_out_flag)
		{
			clear();
		}
		else
		{
			reading_ = k_ * 1000000.0 * float(new_target - 1) / float(durationMicros_);
		}
	}
	else if ((micros() - startMicros_) / 1000.0 > time_out_limit_)
	{
		clear();
	}
}

void Tachometer::incrementEdge()
{
	tachometer_edges_++;
}
