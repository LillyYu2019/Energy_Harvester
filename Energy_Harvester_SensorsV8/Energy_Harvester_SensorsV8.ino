/*
 Energy Harvester Data Recording 
 Justin and Lilly
 2020-01-05
 
 Output the following to be read by MATLAB:
  -PP1
  -PP2
  -Flowrate
  -Torque
  -Turbine speed
  -Voltage
  -Current
  
 Torque sensor : Lorenz DR-2477 (+/-2Nm; +/-5V)
 Motor/generator: Maxon EC motor 496661
 Electric load: 
 GV stepper motor: Nema17
 Gear ratio: 148:18

 towards me is open
 away from me is close
 
 */
 
#include "Tachometer.h"

//Settings
#define decimal_places 4

// DIFINE PIN LOCATION
#define PT1_sensor_pin A2
#define PT2_sensor_pin A1
#define torque_sensor_pin A12
#define V_sensor_pin A14
#define I_sensor_pin A15

#define SPEED_TIMEOUT_MILLIS 500    // Timeout duration before defaulting to zero angular velocity
#define SPEED_EDGES_TO_COUNT 12    // Number of edges to average duration over

#define FLOWRATE_TIMEOUT_MILLIS 500    // Timeout duration before defaulting to zero angular velocity
#define FLOWRATE_EDGES_TO_COUNT 12  // Number of edges to average duration over

// PRESSURE SENSOR VARIABLES
float PT1_sensor_voltage = 0.0;
float PT2_sensor_voltage = 0.0;
float PT1 = 0.0;
float PT2 = 0.0;

//ELECTRIC LOAD VARIABLES
float V_voltage = 0.0;
float I_voltage = 0.0;
float V = 0.0;
float I = 0.0;

// Digital SENSOR VARIABLES
float GPM = 0.0;
float RPM = 0.0;
const float pole_per_rotation = 1.0;
const float K = 246.7;

// TORQUE SENSOR VARIABLES
float torque_sensor_voltage = 0;
float torque = 0.0;

// TIMER VARIABLES
const int num_of_analog_sensors = 5;
float sensor_read_time = 0.0; // in seconds
float start_time = 0.0;
float sample_time_init = 5.0; // in ms
float sample_time[num_of_analog_sensors];

//Initialize tachometer
Tachometer SpeedSensor(PIN_D3, pole_per_rotation, SPEED_TIMEOUT_MILLIS, SPEED_EDGES_TO_COUNT);
Tachometer FlowSensor(PIN_D2, K, FLOWRATE_TIMEOUT_MILLIS, FLOWRATE_EDGES_TO_COUNT);


void setup()
{
    pinMode(PT1_sensor_pin, INPUT);
    pinMode(PT2_sensor_pin, INPUT);
    pinMode(torque_sensor_pin, INPUT);
    pinMode(V_sensor_pin, INPUT);
    pinMode(I_sensor_pin, INPUT);

    for (byte i = 0; i < num_of_analog_sensors; i++)
    {
        sample_time[i] = sample_time_init;
    }
    sample_time[3] = 10.0;

    start_time = millis() / 1000.0;

    Serial.begin(9600);
}

void loop()
{
    //Read all sensors sequencially
    sensor_read_time = (millis() / 1000.0 - start_time);
    PT1_sensor_voltage = analog_read(PT1_sensor_pin, sample_time[1]);
    PT2_sensor_voltage = analog_read(PT2_sensor_pin, sample_time[2]);
    torque_sensor_voltage = analog_read(torque_sensor_pin, sample_time[3]);
    V_voltage = analog_read(V_sensor_pin, sample_time[4]);
    I_voltage = analog_read(I_sensor_pin, sample_time[5]);
    RPM = SpeedSensor.getAngularSpeed(ROT_PER_MINUTE);  //new Exon motor only has one pole, turbine speed in RPM
    GPM = FlowSensor.getAngularSpeed(ROT_PER_MINUTE);  //flowrate in GPM

    // SENSOR mapping of the average reading
    PT1 = analog_mapping(PT1_sensor_voltage, 5, 1, 100, 0);        //psi
    PT2 = analog_mapping(PT2_sensor_voltage, 5, 1, 100, 0);        //psi
    torque = analog_mapping(torque_sensor_voltage, 5, 0, 2000, 0); //mNm
    V = analog_mapping(V_voltage, 5, 0, 30, 0.2);                  //V
    I = analog_mapping(I_voltage, 5, 0, 60, 0.0);                  //Amps                    

    read_input_commends_no_prompt();
    print_to_text();
}

void read_input_commends_no_prompt()
{
    if (Serial.available())
    {
        char c = Serial.read(); //gets one byte from serial buffer
        char temp = Serial.read();
        if (c == 't')
        {
            int sensor_pin = Serial.parseInt();
            char temp = Serial.read();
            float sample = Serial.parseFloat();
            if (sample > 0.0)
                sample_time[sensor_pin] = sample;
        }
    }
}

float analog_read(int port, float sample_time_analog)
{
    float counter = 0.0;
    float sensor_voltage = 0.0;
    long t_current = millis();
    long t_last_read = t_current;

    while ((t_current - t_last_read) < sample_time_analog)
    {
        sensor_voltage += analogRead(port);
        counter += 1.0;
        t_current = millis();
    }
    return sensor_voltage / counter;
}

// Analog sensor mapping
float analog_mapping(float V_out, float V_max, float V_min, float out_max, float out_min)
{
    float measurement = (V_out / 1023.0 * 5.0 - V_min) * (out_max - out_min) / (V_max - V_min) + out_min;
    return measurement;
}

void print_to_text()
{

    Serial.print("t ");
    Serial.println(sensor_read_time, decimal_places);
    Serial.print("PT1 ");
    Serial.println(PT1, decimal_places);
    Serial.print("PT2 ");
    Serial.println(PT2, decimal_places);
    Serial.print("tor ");
    Serial.println(torque, decimal_places);
    Serial.print("V ");
    Serial.println(V, decimal_places);
    Serial.print("I ");
    Serial.println(I, decimal_places);
    Serial.print("RPM ");
    Serial.println(RPM, decimal_places);
    Serial.print("GPM ");
    Serial.println(GPM, decimal_places);
}
