/*
 Energy Harvester Data Recording 
 Lilly
 2020-02-07
 
 Output the following to be read by MATLAB:
  -PP1
  -PP2
  -Flowrate
  -Torque
  -Turbine speed
  -Voltage
  -Current
  -GV angle encoder
  
 Torque sensor : Lorenz DR-2477 (+/-2Nm; +/-5V)
 Motor/generator: Maxon EC motor 496661
 Electric load: 
 GV stepper motor: Nema17 with 99.5 to 1 gearbox
 Gear ratio: 78:18

 towards me is open
 away from me is close
 
 */

#include <Encoder.h>
#include "Tachometer.h"
#include "Analog_sensor.h"

//Global settings
const bool print_sample_time = false;
const int total_num_sensors = 8;
const String sensor_names[] = {"PT1", "PT2", "tor", "V", "I", "RPM", "GPM", "GV"};
const float target_sample_time = 200.0; //ms
const int decimal_places = 4;

//Analog sensor settings
const int num_analog_sensors = 5;
const int analog_target_samples = 120;
const int analog_pins[num_analog_sensors] = {A0, A1, A13, A14, A15};
float analog_range[num_analog_sensors][4] = {
    {5, 1, 100, 0},      //psi
    {5, 1, 100, 0},      //psi
    {5, 0, 2000, 7.885}, //mNm
    {5, 0, 30, 0.0},     //V
    {5, 0, 60, 0.0},    //A
};

//Digital sensor settings
const int num_digital_sensors = 2;
const float digital_time_out_limit = 1000; //ms
const int digital_pins[num_digital_sensors] = {2, 3};
const int digital_target_samples[num_digital_sensors] = {20, 40};
const float digital_k[num_digital_sensors] = {1.0 / 2.0 * 60.0, 1.0 / 2.0 / 246.7 * 60.0};

//Encoder settings
const int num_encoders = 1;
const int encoder_pins[num_encoders][2] = {{18, 19}};
const float encoder_k[num_encoders] = {-1.0 / 4.0 / 600.0 * 200.0 * 1.8 / 99.5075 * 30.0 / 114.0}; // 4x readings, motors has 200 ticks, encoder has 600 ticks

//Global Variables
float start_time;
float sensor_readings[total_num_sensors];
Analog_sensor Analog_sensor_list[num_analog_sensors];
Tachometer Tachometer_list[num_digital_sensors];
Encoder Encoder_list[num_encoders] = {Encoder(encoder_pins[0][0], encoder_pins[0][1])};

void setup()
{
  initialize_all_sensors();

  start_time = millis();

  Serial.begin(9600);
}

void loop()
{
  analog_read();
  digital_read();

  if (millis() - start_time > target_sample_time)
  {
    analog_write();
    digital_write();
    encoder_write();
    print_to_serial();
    start_time = millis();
  }
}

void initialize_all_sensors()
{
  for (byte i = 0; i < total_num_sensors; i++)
  {
    sensor_readings[i] = 0.0;
  }

  for (byte i = 0; i < num_analog_sensors; i++)
  {
    Analog_sensor_list[i] = Analog_sensor(analog_pins[i], analog_target_samples, analog_range[i]);
  }

  for (byte i = 0; i < num_digital_sensors; i++)
  {
    Tachometer_list[i] = Tachometer(digital_k[i], digital_target_samples[i], digital_time_out_limit);
  }

  attachInterrupt(digitalPinToInterrupt(digital_pins[0]), incrementEdge1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(digital_pins[1]), incrementEdge2, CHANGE);
}

void analog_read()
{
  for (byte i = 0; i < num_analog_sensors; i++)
  {
    Analog_sensor_list[i].read();
  }
}

void analog_write()
{
  for (byte i = 0; i < num_analog_sensors; i++)
  {
    sensor_readings[i] = Analog_sensor_list[i].get_readings();
  }
}

void digital_read()
{
  for (byte i = 0; i < num_digital_sensors; i++)
  {
    Tachometer_list[i].read();
  }
}

void digital_write()
{
  for (byte i = 0; i < num_digital_sensors; i++)
  {
    sensor_readings[i + num_analog_sensors] = Tachometer_list[i].get_readings();
  }
}

void encoder_write()
{
  for (byte i = 0; i < num_encoders; i++)
  {
    sensor_readings[i + num_analog_sensors + num_digital_sensors] = Encoder_list[i].read() * encoder_k[i];
  }
}

void incrementEdge1()
{
  Tachometer_list[0].increment_edge();
}

void incrementEdge2()
{
  Tachometer_list[1].increment_edge();
}

void print_to_serial()
{
  Serial.print("t ");
  Serial.println(millis() / 1000.0, decimal_places);
  for (byte i = 0; i < total_num_sensors; i++)
  {
    Serial.print(sensor_names[i] + " ");
    Serial.println(sensor_readings[i], decimal_places);
  }
  if (print_sample_time == true)
    {
      Serial.print("analog duration ");
      Serial.println(Analog_sensor_list[0].get_sample_time(), decimal_places);
      for (byte i = 0; i < num_digital_sensors; i++)
      {
        Serial.print(sensor_names[i + num_analog_sensors] + " duration ");
        Serial.println(Tachometer_list[i].get_sample_time(), decimal_places);
      }
    }
}
