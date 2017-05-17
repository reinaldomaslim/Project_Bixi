#include <ros.h>
#include <std_msgs/Bool.h>
#include <SoftwareSerial.h>

// stepper motor constants
const int stepper_speed = 80; // speed 0-100
const int PUp = 2;
const int PUn = 3;
const int DRp = 4;
const int DRn = 5;
const int MFp = 7;
// const int MFn = 8;


// sensors
const int sensor1 = 8;
const int sensor2 = 9;
const int limit_switch1 = A5;
const int limit_switch2 = 41;
const int power1 = A8;
const int power2 = 33;


std_msgs::Bool msg;
std_msgs::Bool sense;

ros::NodeHandle nh;
ros::Publisher limit_sense("limit_sense", &sense);
ros::Publisher job_status("job_status", &msg);

SoftwareSerial mySerial(12,13);
int myTarget = 0; // target position, 0-4095 is the range of the JRK21V3 controller. 
// 0 - extend at max speed, 4095 - retract at max speed, 2048 - stop
int linear_wait_time = 5500; // in ms
float y = map(stepper_speed, 0, 100, 1000, 230);

void linear_motor(bool a)
{
  word target;
  if (a == true) target = 0;
  else target = 4095;
  //  word target = x;  //only pass this ints, i tried doing math in this and the remainder error screwed something up
  mySerial.write(0xAA); //tells the controller we're starting to send it commands
  mySerial.write(0xB);   //This is the pololu device # you're connected too that is found in the config utility(converted to hex). I'm using #11 in this example
  mySerial.write(0x40 + (target & 0x1F)); //first half of the target, see the pololu jrk manual for more specifics
  mySerial.write((target >> 5) & 0x7F);   //second half of the target, " " " 
  delay(linear_wait_time);
}

void stepper_motor(int x)
{
  if (x == 0)
  {
    digitalWrite(DRn, HIGH);
    delayMicroseconds(10);
    while (digitalRead(sensor2) == LOW)
    {
      digitalWrite(PUn, HIGH);
      delayMicroseconds(y);
      digitalWrite(PUn, LOW);
      delayMicroseconds(y);
    }
  }
  else if (x == 1)
  {
    digitalWrite(DRn, LOW);
    delayMicroseconds(10);
    while (digitalRead(sensor1) == LOW)
    {
      digitalWrite(PUn, HIGH);
      delayMicroseconds(y);
      digitalWrite(PUn, LOW);
      delayMicroseconds(y);
    }
  }
  else if (x == 2) // go down a bit
  {
    digitalWrite(DRn, HIGH);
    int pose_motor = 0;
    delayMicroseconds(10);
    while (pose_motor < 2000)
    {
      digitalWrite(PUn, HIGH);
      delayMicroseconds(y);
      digitalWrite(PUn, LOW);
      delayMicroseconds(y);
      pose_motor++;
    }
  }
  
}


void setup()
{
  // ros initialize
  nh.initNode();
  nh.advertise(job_status);
  nh.advertise(limit_sense);
  
  // initialize cmd with JRK21V3 
  mySerial.begin(9600);
  
  // initialize stepper motor
  pinMode(PUp, OUTPUT);
  pinMode(PUn, OUTPUT);
  pinMode(DRp, OUTPUT);
  pinMode(DRn, OUTPUT);
  pinMode(MFp, OUTPUT);
  // pinMode(MFn, OUTPUT);
  pinMode(sensor1, INPUT);
  pinMode(sensor2, INPUT);
  pinMode(limit_switch1, INPUT);
  pinMode(limit_switch2, INPUT);
  pinMode(power1, OUTPUT);
  pinMode(power2, OUTPUT);
  digitalWrite(power1, HIGH);
  digitalWrite(power2, HIGH);
  
  digitalWrite(PUp, HIGH);
  digitalWrite(DRp, HIGH);
  // digitalWrite(DRn, LOW); // low - up, high - down
  digitalWrite(MFp, HIGH);
  // digitalWrite(MFn, LOW);
  digitalWrite(PUn, HIGH);

  Serial.begin(9600);


  linear_motor(1); // extend linear motor
  stepper_motor(1); // stepper top
}

void loop()
{
  if (digitalRead(limit_switch1) && digitalRead(limit_switch2))
  {
    sense.data = true;
    limit_sense.publish(&sense);
    delay(1000);
    
    stepper_motor(2); // stepper goes down a bit
    linear_motor(0);  // retract linear actuator
    stepper_motor(0); // stepper goes all the way down
    linear_motor(1);  // extend linear actuator
    stepper_motor(1); // stepper goes all the way up

    msg.data = true;
    job_status.publish(&msg);
  }
  delay(100);
}




