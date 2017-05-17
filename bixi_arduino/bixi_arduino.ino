#include <ros.h>
#include <std_msgs/Bool.h>
#include <SoftwareSerial.h>

std_msgs::Bool cmd;           // local cmd from tx1
std_msgs::Bool msg; // to publish back to tx1
SoftwareSerial mySerial(7,8);
int myTarget = 0; // target position, 0-4095 is the range of the JRK21V3 controller. 
// 0 - extend at max speed, 4095 - retract at max speed, 2048 - stop


ros::NodeHandle nh;
// declaring publisher and subscriber
ros::Publisher to_ros_chatter("to_ros_chatter", &msg);


// ros message call back
void messageCb(const std_msgs::Bool& chat)
{
  msg.data = cmd.data = chat.data;
  to_ros_chatter.publish( &msg );
}

ros::Subscriber<std_msgs::Bool> sub("to_arduino_chatter", &messageCb);

//sets the new target for the JRK21V3 controller, this uses pololu high resulution protocal
void Move(int x) {
  word target = x;  //only pass this ints, i tried doing math in this and the remainder error screwed something up
  mySerial.write(0xAA); //tells the controller we're starting to send it commands
  mySerial.write(0xB);   //This is the pololu device # you're connected too that is found in the config utility(converted to hex). I'm using #11 in this example
  mySerial.write(0x40 + (target & 0x1F)); //first half of the target, see the pololu jrk manual for more specifics
  mySerial.write((target >> 5) & 0x7F);   //second half of the target, " " " 
}  



void setup()
{
  // initialize ros
  nh.initNode();
  nh.advertise(to_ros_chatter);
  nh.subscribe(sub);

  // initialize cmd with JRK21V3 
  mySerial.begin(9600);
}


void loop()
{
  if (cmd.data == true) myTarget = 0;
  else myTarget = 4095;
  
  Move(myTarget);
  nh.spinOnce();
  delay(100);
}

