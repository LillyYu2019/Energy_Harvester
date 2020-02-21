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
}

float Tachometer::get_readings()
{
	return reading_;
}

float Tachometer::get_sample_time()
{
	return duration_micros_ / 1000.0;
}

void Tachometer::clear()
{
	reading_ = 0.0;
	duration_micros_ = 0.0;
	start_micros_ = micros();
}

void Tachometer::read()
{
	if ((micros() - start_micros_) / 1000.0 > time_out_limit_)
	{
		clear();
		return;
	}

	if (tachometer_edges_ >= target_sample_count_)
	{
		int new_target = tachometer_edges_ + 1;
		bool timer_stopped = false;

		while ((micros() - start_micros_) / 1000.0 < time_out_limit_ && !timer_stopped)
		{
			if (tachometer_edges_ >= new_target)
			{
				float current_time = micros();
				duration_micros_ = current_time - start_micros_;
				start_micros_ = current_time;
				tachometer_edges_ = 1;
				timer_stopped = true;
			}
		}

		if (timer_stopped)
		{
			reading_ = k_ * 1000000.0 * float(new_target - 1) / float(duration_micros_);
		}
		else
		{
			clear();
		}
	}
}

void Tachometer::increment_edge()
{
	tachometer_edges_++;
}
