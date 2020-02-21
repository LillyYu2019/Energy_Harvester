/******************************************************************************
	Analog_sensor.h
	Library for reading averaged analog signals
	Lilly Yu @ WERL, U of Toronto (adapted from Rowan Walsh's version)
	2020-02-07


******************************************************************************/

#include "Arduino.h"
#include "Analog_sensor.h"

Analog_sensor::Analog_sensor() {}

Analog_sensor::Analog_sensor(int pin, int analog_target_samples, float *analog_range)
{
    pin_ = pin;
    target_sample_count_ = analog_target_samples;
    analog_range_ = analog_range;

    reading_ = 0.0;
    sample_time_ = 0.0;

    clear();

    pinMode(pin_, INPUT);
}

void Analog_sensor::clear()
{
    read_count_ = 0;
    voltage_ = 0.0;
    start_time_ = millis();
}

void Analog_sensor::map()
{
    float measurement = voltage_ / (float)read_count_ / 1023.0 * 5.0;
    reading_ = analog_range_[3] +
               (analog_range_[2] - analog_range_[3]) * (measurement - analog_range_[1]) / (analog_range_[0] - analog_range_[1]);
    sample_time_ = millis() - start_time_;
}

float Analog_sensor::get_readings()
{
    return reading_;
}

float Analog_sensor::get_sample_time()
{
    return sample_time_;
}

void Analog_sensor::read()
{
    voltage_ += analogRead(pin_);
    read_count_ += 1;

    if (read_count_ > target_sample_count_)
    {
        map();
        clear();
    }
}
