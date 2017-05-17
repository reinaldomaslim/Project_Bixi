#include <ros.h>
#include <std_msgs/Bool.h>
#include <SoftwareSerial.h>

// stepper motor constants
const int stepper_speed = 80; // speed 0-100
const int PUp = 2;
const int PUn = 3;
const int DRp = 4;
const int DRn = 5;
const int MFp = 6;
const int MFn = 7;

// sensors
const int sensor1 = 9;
const int sensor2 = 10;

// rx_tx pins
const int rx_pin = 0;
const int tx_pin = 7;

SoftwareSerial mySerial(12,13);
int myTarget = 0; // target position, 0-4095 is the range of the JRK21V3 controller. 
// 0 - extend at max speed, 4095 - retract at max speed, 2048 - stop
int linear_wait_time = 1000; // in ms

float y = map(stepper_speed, 0, 100, 1000, 230);
bool start=false;
bool finish=false;

void linear_motor(bool x)
{
  word target;
  if (x == true) target = 0;
  else target = 4095;
  //  word target = x;  //only pass this ints, i tried doing math in this and the remainder error screwed something up
  mySerial.write(0xAA); //tells the controller we're starting to send it commands
  mySerial.write(0xB);   //This is the pololu device # you're connected too that is found in the config utility(converted to hex). I'm using #11 in this example
  mySerial.write(0x40 + (target & 0x1F)); //first half of the target, see the pololu jrk manual for more specifics
  mySerial.write((target >> 5) & 0x7F);   //second half of the target, " " " 
  delay(linear_wait_time);
}

void setup()
{
  // initialize cmd with JRK21V3 
  mySerial.begin(9600);
  
  // initialize stepper motor
  const int val = 80; // speed 0-100
  const int PUp = 2;
  const int PUn = 3;
  const int DRp = 4;
  const int DRn = 5;
  const int MFp = 7;
  const int MFn = 8;
  digitalWrite(PUp, HIGH);
  digitalWrite(DRp, HIGH);
  // digitalWrite(DRn, LOW); // low - up, high - down
  digitalWrite(MFp, HIGH);
  digitalWrite(MFn, LOW);
  digitalWrite(PUn, HIGH);

  Serial.begin(9600);
  pinMode(tx_pin, OUTPUT);
  pinMode(rx_pin, INPUT);
  digitalWrite(tx_pin, LOW);
}



void loop()
{
  start=digitalRead(rx_pin);
  
  if ( start == true)
  {
    //start actuation
//    linear_motor(0);
//    linear_motor(1);
    delay(2000);  
    //finished actuation
    finish=true;
  }
  else{
    finish=false; 
  }
  
  if(finish==true)
  {
    digitalWrite(tx_pin, HIGH);
    finish=false;
  }
  else{
    digitalWrite(tx_pin, LOW);
  }
  
  delay(100); 
  

//  if (digitalRead(rx_pin) == 1)
//  {
//    Serial.println("HIGH!");
//  }
//  else Serial.println("LOW");
//  delay(100);


//  digitalWrite(tx_pin, HIGH);
//  delay(1000);
//  digitalWrite(tx_pin, LOW);
//  delay(1000);
}

