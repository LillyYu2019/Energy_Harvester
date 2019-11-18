/*
 Energy Harvester Data Recording 
 Justin and Lilly
 2019-11-15
 
 Output the following to be read by MATLAB:
  -PP1
  -PP2
  -PT1
  -PT2
  -Flowrate
  -Turbine RPM
  -Turbine TOrque
  
 Torque sensor : Lorenz DR-2477 (+/-2Nm; +/-5V)
 Motor/generator: Maxon EC motor 496661
 Electric load: 
 GV stepper motor: Nema17
 Gear ratio: 148:18

 towards me is open
 away from me is close
 
 */

#include <HalfStepper.h> //https://github.com/FancyFoxGems/HalfStepper https://www.arduinolibraries.info/libraries/half-stepper

//Settings

bool print_to_screen = false;

// DIFINE PIN LOCATION
#define DP1_sensor_pin A0
#define PT1_sensor_pin A1
#define PT2_sensor_pin A2
#define torque_sensor_pin A3
#define flow_sensor_pin 2
#define RPM_sensor_pin 3
#define in1Pin 11
#define in2Pin 10
#define in3Pin 9
#define in4Pin 8
#define V_sensor_pin A4
#define I_sensor_pin A5
#define CC_CV_commend_pin 7

//Initialize stepper motor
HalfStepper motor(200, in1Pin, in2Pin, in3Pin, in4Pin); //Nema17 stepper motor
int GV_angle = 0;

// PRESSURE SENSOR VARIABLES
float DP1_sensor_voltage = 0.0;
float PT1_sensor_voltage = 0.0;
float PT2_sensor_voltage = 0.0;
float DP1 = 0.0;
float PT1 = 0.0;
float PT2 = 0.0;

//ELECTRIC LOAD VARIABLES
float V_voltage = 0.0;
float I_voltage = 0.0;
float V = 0.0;
float I = 0.0;

// Digital SENSOR VARIABLES
float flow_counter = 0.0;
float RPM_counter = 0.0;
float GPM = 0.0;
float RPM = 0.0;
const float K = 246.7;

// TORQUE SENSOR VARIABLES
float torque_sensor_voltage = 0;
float torque = 0.0;

// TIMER VARIABLES
const float sample_time_analog = 100.0; // in ms
const float sample_time_digital = 100.0;   // in ms
float sensor_read_time = 0.0; // in seconds
float start_time = 0.0;


void setup() {
  
  pinMode(DP1_sensor_pin, INPUT);
  pinMode(PT1_sensor_pin, INPUT);
  pinMode(PT2_sensor_pin, INPUT);
  pinMode(flow_sensor_pin, INPUT);
  pinMode(RPM_sensor_pin,INPUT);
  pinMode(torque_sensor_pin,INPUT);
  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);
  pinMode(in3Pin, OUTPUT);
  pinMode(in4Pin, OUTPUT);
  pinMode(V_sensor_pin, INPUT);
  pinMode(I_sensor_pin, INPUT);
  pinMode(CC_CV_commend_pin, OUTPUT);

  start_time = millis()/1000.0;
  
  motor.setSpeed(5);
  
  Serial.begin (9600);
}


void loop() {
  
  //Read all sensors sequencially
  sensor_read_time = (millis()/1000.0 - start_time);
  DP1_sensor_voltage = analog_read(DP1_sensor_pin);
  PT1_sensor_voltage = analog_read(PT1_sensor_pin);
  PT2_sensor_voltage = analog_read(PT2_sensor_pin);
  torque_sensor_voltage = analog_read(torque_sensor_voltage);
  V_voltage = analog_read(V_sensor_pin);
  I_voltage = analog_read(I_sensor_pin);
  RPM_counter = digital_read(RPM_sensor_pin);
  flow_counter = digital_read(flow_sensor_pin);
  
  // SENSOR mapping of the average reading
  DP1 = analog_mapping(DP1_sensor_voltage, 5, 0, 30, 0); //psi
  PT1 = analog_mapping(PT1_sensor_voltage, 5, 1, 100, 0);//psi
  PT2 = analog_mapping(PT2_sensor_voltage, 5, 1, 100, 0);//psi
  torque = analog_mapping(torque_sensor_voltage, 5, 0, 2000, 0);//mNm
  V = analog_mapping(V_voltage, 5, 0, 30, 0.2); //V
  I = analog_mapping(I_voltage, 5, 0, 60, 0.2); //Amps
  RPM = RPM_counter / 2.0 * 60.0;  //new Exon motor only has one pole, turbine speed in RPM
  GPM = flow_counter / K / 2.0 * 60.0; //flowrate in GPM

  if (print_to_screen){
    read_input_commends_with_prompt();
    print_to_monitor();
  }else{
    read_input_commends_no_prompt();
    print_to_text();
  }
  
}

//Read input commends
void read_input_commends_with_prompt(){
  if (Serial.available()){
    char c = Serial.read();  //gets one byte from serial buffer
    char temp = Serial.read();
    if (c == 'G'){
      Serial.println("Please enter GV angle, pos is cose, neg is open");
      int deg = Serial.parseInt(); // pos is close, neg is open
      while (deg <= 0){
        deg = Serial.parseInt();
      }
      move_GV(deg);
      }
     else if (c == 'I'){
      Serial.println("please enter desired current (0 to 6A)");
      float cur = Serial.parseFloat();
      while (cur <= 0.0){
        cur = Serial.parseFloat();
      }
      Serial.println(cur);
      set_current(cur);
     }
     else if (c =='V'){
      Serial.println("Please enter desired voltage (0 to 30V)");
      float vol = Serial.parseFloat();
      while (vol <= 0.0){
        vol = Serial.parseFloat();
      }
      Serial.println(vol);
      set_voltage(vol);
     }
     else if (c!='\n'){
      Serial.println("Please enter G for GV, I for current, or V for voltage settings");
     }
  }
 }

 void read_input_commends_no_prompt(){
  if (Serial.available()){
    char c = Serial.read();  //gets one byte from serial buffer
    char temp = Serial.read();
    if (c == 'G'){
      int deg = Serial.parseInt(); // pos is close, neg is open
      while (deg <= 0){
        deg = Serial.parseInt();
      }
      move_GV(deg);
      }
     else if (c == 'I'){
      float cur = Serial.parseFloat();
      while (cur <= 0.0){
        cur = Serial.parseFloat();
      }
      set_current(cur);
     }
     else if (c =='V'){
      float vol = Serial.parseFloat();
      while (vol <= 0.0){
        vol = Serial.parseFloat();
      }
      set_voltage(vol);
     }
  }
 }

// Stepper motor commend to move GVs
void move_GV (int deg){
    float deg_ratio = deg*148.0/18.0; //big gear #teeth = 148, small gear #teeth = 18
    float steps_gear = deg_ratio/0.9; // half stepping
    GV_angle = GV_angle + deg;
    motor.step(steps_gear);
}

void set_current(float cur){
  int commend = cur/60.0*255.0;
  analogWrite(CC_CV_commend_pin, commend);
}

void set_voltage(float vol){
  int commend = vol/30.0*255.0;
  analogWrite(CC_CV_commend_pin, commend);
}

// Digital SENSOR read
float digital_read(int port){
  
  float counter = 0.0;
  long t_current=millis();
  long t_last_read=t_current;
  bool current_state = digitalRead(port);
  bool last_state = current_state;
  
  while ((t_current-t_last_read) < sample_time_digital){
    current_state = digitalRead(port);
    if (current_state!=last_state){
    counter+= 1.0;
    last_state = current_state;
    }
    t_current=millis();
  }
  return counter /sample_time_digital*1000.0;
 }

float analog_read(int port){
  
  float counter = 0.0;
  float sensor_voltage = 0.0;
  long t_current=millis();
  long t_last_read=t_current;
  
  while ((t_current-t_last_read) < sample_time_analog){
    sensor_voltage += analogRead(port);
    counter+= 1.0;
    t_current=millis();
  }
  return sensor_voltage/counter;
 }

// Analog sensor mapping
float analog_mapping(float V_out, float V_max, float V_min, float out_max, float out_min ){
  float measurement = (V_out/1023.0*5.0 - V_min)*(out_max-out_min)/ (V_max-V_min) + out_min;
  return measurement;
}

void print_to_monitor(){
  
    Serial.print("time: ");
    Serial.println(sensor_read_time,4);
    Serial.print("DP1: ");
    Serial.println(DP1,4);
    Serial.print("PT1: ");
    Serial.println(PT1,4);
    Serial.print("PT2: ");   
    Serial.println(PT2,4); 
    Serial.print("torque: " );
    Serial.println(torque,4);
    Serial.print("V: ");   
    Serial.println(V,4);
    Serial.print("I: ");   
    Serial.println(I,4); 
    Serial.print("RPM: ");
    Serial.println(RPM,4);
    Serial.print("GPM: ");  
    Serial.println(GPM,4);
    Serial.print("GV angle:");   
    Serial.println(GV_angle);      
    Serial.println( );                 
}

void print_to_text(){

    Serial.println(sensor_read_time,4);
    Serial.println(DP1,4);
    Serial.println(PT1,4);
    Serial.println(PT2,4);
    Serial.println(torque,4); 
    Serial.println(V,4);
    Serial.println(I,4);  
    Serial.println(RPM,4);
    Serial.println(GPM,4);
    Serial.println(GV_angle);              
}

  
