

/*
 Energy Harvester Stepper motor control
 Justin and Lilly
 2020-02-04
 
 GV stepper motor: Nema17
 Gear ratio: 148:18

 GV stepper motor: Nema17 with gear box
 outer Gear ratio: 78:18
 inter gear box ratio: 99 1044/2057:1
 
 towards me is open
 away from me is close
 
 */
#include <HalfStepper.h>
#include <Stepper.h>

#define in1Pin 11
#define in2Pin 10
#define in3Pin 9
#define in4Pin 8

//Initialize stepper motor
Stepper motor(200, in1Pin, in2Pin, in3Pin, in4Pin); //Nema17 stepper motor
float GV_angle = 3.0;
float motor_speed = 50.0;
float sensor_read_time = 0.0; // in seconds
float start_time = 0.0;

void setup() {
  
    pinMode(in1Pin, OUTPUT);
    pinMode(in2Pin, OUTPUT);
    pinMode(in3Pin, OUTPUT);
    pinMode(in4Pin, OUTPUT);

    motor.setSpeed(motor_speed);

    start_time = millis() / 1000.0;

    Serial.begin(9600);
}

void loop() {
  read_input_commends_with_prompt();
}

void read_input_commends_with_prompt()
{
    if (Serial.available())
    {
        char c = Serial.read(); //gets one byte from serial buffer
        char temp = Serial.read();
        if (c == 'G')
        {
            Serial.println("Please enter GV angle, pos is cose, neg is open");
            float deg = Serial.parseFloat(); // pos is close, neg is open
            while (deg <= 0.1 && deg > -0.1)
            {
                deg = Serial.parseFloat();
            }
            sensor_read_time = (millis() / 1000.0 - start_time);
            print_to_monitor();
            move_GV(deg);
            sensor_read_time = (millis() / 1000.0 - start_time);
            print_to_monitor();
            Serial.println();
        }
        else if (c == 'S')
        {
          Serial.println("Please enter GV speed (0 to 20): ");
          float m_speed = Serial.parseFloat();
          while (m_speed <= 0.1 || m_speed > 30.0)
            {
                m_speed = Serial.parseFloat();
            }
            motor.setSpeed(m_speed);
        }
        else if (c != '\n')
        {
          Serial.println(c);
          Serial.println("Please enter G for GV, S for motor speed settings");
        }
    }
}

void move_GV(float deg)
{
    float deg_ratio = deg * 78.0 / 18.0 * 99.5075; //for NEMA17 with gearbox: 1 deg = 239.555 steps
    float steps_gear = deg_ratio / 1.8;  
    Serial.print("steps taken: ");
    Serial.println(steps_gear, 2);
    GV_angle = GV_angle + deg;
    motor.step(steps_gear);
}

void print_to_monitor()
{

    Serial.print("time: ");
    Serial.println(sensor_read_time, 4);
    Serial.print("GV angle:");
    Serial.println(GV_angle);

}
