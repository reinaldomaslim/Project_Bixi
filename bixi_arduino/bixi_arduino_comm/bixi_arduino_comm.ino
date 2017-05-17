#include <ros.h>
#include <std_msgs/Bool.h>

const int tx_pin = 8;
const int rx_pin = 10;


std_msgs::Bool msg;

ros::NodeHandle nh;
ros::Publisher to_ros_chatter("to_ros_chatter", &msg);

std_msgs::Bool cmd;

bool start=false;
bool finish=false;

void messageCb(const std_msgs::Bool& chat)
{
  start = chat.data;
}

ros::Subscriber<std_msgs::Bool> sub("to_arduino_chatter", &messageCb);



void setup()
{
  nh.initNode();
  nh.advertise(to_ros_chatter);
  nh.subscribe(sub);
  pinMode(tx_pin, OUTPUT);
  pinMode(rx_pin, INPUT);
  
}

void loop()
{
  
  if (start == true)
  {
    digitalWrite(tx_pin, HIGH);
    delay(2000);
  }
  else 
  {
    digitalWrite(tx_pin, LOW);
    delay(100);
  }
  
  
  if (digitalRead(rx_pin) == true)
  {
    finish = true;
  }
  
  cmd.data = finish;
  to_ros_chatter.publish(&cmd);
  finish = false;
    
  nh.spinOnce();
  delay(100);

//  digitalWrite(tx_pin, HIGH);
//  delay(1000);
//  digitalWrite(tx_pin, LOW);
//  delay(1000);
}

