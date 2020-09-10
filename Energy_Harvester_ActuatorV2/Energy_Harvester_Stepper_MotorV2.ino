

/*
 Energy Harvester Stepper motor control
 Justin and Lilly
 2020-08-06
 
 GV stepper motor: Nema17
 Gear ratio: 148:18

 GV stepper motor: Nema17 with gear box
 outer Gear ratio: 78:18
 inter gear box ratio: 99 1044/2057:1
 
 towards me is open
 away from me is close
 
 */
 
#include <Stepper.h>
#include <SPI.h>
#include <Encoder.h>
#include "DAC_MCP49xx.h"

#define in1Pin 7
#define in2Pin 6
#define in3Pin 5
#define in4Pin 4
#define SS_PIN 10
#define I_pin A0

float I_last_time = 0.0; // in seconds
float current_time = 0.0;
float print_time = 0.0;
float last_print_time = 0.0;
int state = 3;
int last_state = 1;

//control cmd
float I_des = 2.0;
float g_des = 0.0;
float I_speed = 0.05; // A/sec

//move current
float I_signal = 0.0;
const float I_offset = -0.03;
const float I_min = 1.0;
const float I_max = 3.4;
const float g_min = 0.0;
const float g_max = 8.4;

//read current
float I = 0.0;
float analog_range[4] = {5, 0, 60, 0.0};
int analog_target_samples = 500;
float voltage = 0.0;
int read_count = 0;

//Move guidevane
Stepper motor(200, in1Pin, in2Pin, in3Pin, in4Pin); //Nema17 stepper motor
float motor_speed = 60.0;

//read guide vane
Encoder gv_angle(3,2);
float g = 0.0;
float encoder_k = -1.0 / 4.0 / 600.0 * 200.0 * 1.8 / 99.5075 * 30.0 / 114.0;// 4x readings, motors has 200 ticks, encoder has 600 ticks

DAC_MCP49xx dac(DAC_MCP49xx::MCP4921, SS_PIN);

void setup()
{

    pinMode(in1Pin, OUTPUT);
    pinMode(in2Pin, OUTPUT);
    pinMode(in3Pin, OUTPUT);
    pinMode(in4Pin, OUTPUT);

    motor.setSpeed(motor_speed);

    I_last_time = millis() / 1000.0;
    last_print_time = millis() / 1000.0;

    dac.setSPIDivider(SPI_CLOCK_DIV16);
    dac.setPortWrite(true);
    dac.output(0);

    Serial.begin(9600);
}

void loop()
{

    read_input_commends_with_prompt();
    actuate();
}

void read_input_commends_with_prompt()
{
  if (Serial.available())
  {
    char c = Serial.read(); //gets one byte from serial buffer
    if (c == 'C')
    {
      float curr = Serial.parseFloat();
      float d = Serial.parseFloat(); // pos is close, neg is open
      
      if (curr > I_min - 0.1 && curr < I_max + 0.1 && d > g_min - 0.1 && d < g_max + 0.1){ //&& 

        if (curr < 0.0142*d*d+0.0177*d+0.8){
          I_des =  0.0142*d*d+0.0177*d+0.8;
          g_des = d;
          Serial.print("Ides too low");
          Serial.print(I_des);
        }
        else{
          I_des = curr;
          g_des = d;
        }
        
        last_state = state;
        state = 0;
      }     
    }
  }
}

void actuate()
{
  g = gv_angle.read() * encoder_k;
  I = I_get_readings() + I_offset;

  if (state == 0)
  { 
    if (g_des > g && I_des > I)
      state = 2;
    else if (g_des < g && I_des < I)
      state = 1;
    else
    {
      if (last_state == 1)
        state = 2;
      else
        state = 1;       
    }
  }

  if (state == 1)
  {
    float diff = g_des - g;
    if (abs(diff) < 0.1)
    {
      state = 2;
//      Serial.println(g);
    }
    else{
      move_GV(diff);
    }
  }

  if (state == 2)
  {
    float diff = I_des - I;
    if (abs(diff) < 0.1)
    {
      state = 1;
//      Serial.println(I);
    }
    else{
      current_time = millis() / 1000.0;
      if (current_time - I_last_time > 0.25){
        
        if (diff > 0)
          I_signal += min(diff, I_speed);
        else
          I_signal += max(diff, -1*I_speed);

        if (I_signal >= 0 && I_signal <= I_max - I_min)
          set_current(I_signal);
        else
          I_signal = 0;
        
        I_last_time = current_time;
      }
    }
  }
}

void move_GV(float deg)
{
    float deg_ratio = deg * 78.0 / 18.0 * 99.5075; //for NEMA17 with gearbox: 1 deg = 239.555 steps
    float steps_gear = deg_ratio / 1.8;
    motor.step(steps_gear);
//    Serial.print("moving GV ");
//    Serial.println(deg);
}
void set_current(float cur)
{
    int commend = (cur+0.015) / 60.0 * 4095.0;
    dac.output(commend);
//    Serial.print("moving I ");
//    Serial.println(I_signal);
}

float I_get_readings()
{
  for (int i = 0; i<analog_target_samples;i++)
    {
      voltage += analogRead(I_pin);
      read_count += 1;
    }

    float measurement = voltage / (float)read_count/ 1023.0 * 5.0;
    float reading = analog_range[3] +
               (analog_range[2] - analog_range[3]) * (measurement - analog_range[1]) / (analog_range[0] - analog_range[1]);
    voltage = 0.0;
    read_count = 0.0;

    return reading;
}
