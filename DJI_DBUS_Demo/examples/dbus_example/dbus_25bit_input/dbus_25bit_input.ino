/*
 This program enables the reading of DBUS 25 bits input 
 and output to DBUS 18 bit or 25 bit

  reading result = dBus.channels[0-7]

*/


#include "DJI_DBUS.h"
#include <Servo.h> 
#include "FlexiTimer2.h" 

//there are two versions of DJI controller protocol
// one is the latestest firmware with 25 bits with 7 channel
//another is the old one with 6 channel of 18 bits
//please use America coding methods
/*
 * *
for each data 
mid value = bin 10000000000   (1024)   0x400
max value = bin 11010010100   (1684)   0x694
max value = bin 00101101100   (364)   0x16C
 */
void write18BitsDbusData();
void write25BitsDbusData();
boolean running = false;
const byte channel = 7;
//#define __6_CHANNELS__
#ifdef __6_CHANNELS__
//channel define
  #define LEFT_UD 3
  #define LEFT_LR 2
  #define RIGHT_UD 1
  #define RIGHT_LR 0
  #define CHANNEL_L 5
  #define CHANNEL_R 4
  #define CHANNEL_UP 1
  #define CHANNEL_MID 3
  #define CHANNEL_DOWN 2
  static uint16_t DBus_Output[channel]  = {1024, 1024, 1024, 1024, CHANNEL_MID, CHANNEL_MID, 1024};
#else
  #define LEFT_UD 2
  #define LEFT_LR 3
  #define RIGHT_UD 1
  #define RIGHT_LR 0
  #define CHANNEL_L 5
  #define CHANNEL_R 6
  #define CHANNEL_3 4
  #define CHANNEL_UP 1568
  #define CHANNEL_MID 1024
  #define CHANNEL_DOWN 511
  static uint16_t DBus_Output[channel]  = {1024, 1024, 1024, 1024, 1024, CHANNEL_MID, CHANNEL_MID};
#endif
//the value of channel can only be 1-3 , the sequence is 01,11,10
//DBus_Output[LEFT_U] is the value of left up and down


DJI_DBUS dBus(Serial1);
static uint16_t rcValue[channel]  = {1684, 1024, 1024, 1024, 1024, 1024, 1024};
int led = 13; 
uint8_t i;

void setup(){
  pinMode(led, OUTPUT);
  Serial.begin(115200);
//  Serial.println("DBUS Status");
  Serial2.begin(100000,SERIAL_8E1);
  FlexiTimer2::set(7, writeDataTimer);        // 中断设置函数，每 500ms 进入一次中断
  FlexiTimer2::start();                //开始计时

  dBus.begin();
  Serial.println("Initialized");
}

void loop(){
  
  if (dBus.Failsafe() == DBUS_SIGNAL_OK) digitalWrite(led, HIGH);
  dBus.FeedLine();
  digitalWrite(led, LOW);

  if (dBus.toChannels == 1){
    dBus.UpdateChannels();
    dBus.toChannels = 0;

    for(i=0;i<channel;i++){
      DBus_Output[i] = dBus.channels[i];
    }

    
  }

  
}

void writeDataTimer()                        //中断处理函数，改变灯的状态
{                       
  write25BitsDbusData();
}
void printDBUSStatus()
{
  Serial.print("Thr ");
  Serial.print(dBus.channels[2]);
  Serial.print(" Ail ");
  Serial.print(dBus.channels[0]);
  Serial.print(" Ele ");
  Serial.print(dBus.channels[1]);
  Serial.print(" Rud ");
  Serial.print(dBus.channels[3]);
  Serial.print(" Fm ");
  Serial.print(dBus.channels[6]);
  Serial.print(" IOC ");
  Serial.print(dBus.channels[5]);
  Serial.print(" Pit  ");
  Serial.print(dBus.channels[4]);
  Serial.print(" Stat ");
  if (dBus.Failsafe() == DBUS_SIGNAL_FAILSAFE) {
    Serial.print("FailSafe");
  } else if (dBus.Failsafe() == DBUS_SIGNAL_LOST) {
    Serial.print("Signal Lost");
  } else if (dBus.Failsafe() == DBUS_SIGNAL_OK) {
    Serial.print("OK");
  }
  Serial.println(".");
  
}

void write18BitsDbusData(){
  //05 04 20 00 01 D8 00 00 00 00 00 00 00 00 00 00 00 00 original 
  //00 04 20 00 01 58 00 00 00 00 00 00 00 00 00 00 00 00 adjusted

  Serial2.write((uint8_t) ( ((DBus_Output[0]&0x00FF)>>0) ) );//data1 0-7
  Serial2.write((uint8_t) ( ((DBus_Output[0]&0x0700)>>8) | ((DBus_Output[1]&0x001F)<<3) ) );//data1 8-10 data2 0-4 
  Serial2.write((uint8_t) ( ((DBus_Output[1]&0x07E0)>>5) | ((DBus_Output[2]&0x0003)<<6) ) );// data2 5-10 data3 0-1
  Serial2.write((uint8_t) ( ((DBus_Output[2]&0x03FC)>>2) ) );// data3 2-9
  Serial2.write((uint8_t) ( ((DBus_Output[2]&0x0400)>>10) | ((DBus_Output[3]&0x007F)<<1) ) );// data3 10 data 4 0-6
  Serial2.write((uint8_t) ( ((DBus_Output[3]&0x0780)>>7) | ((DBus_Output[4]&0x0003)<<4) | ((DBus_Output[5]&0x0003)<<6) ) );// data3 10 data 4 0-6
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
  Serial2.write(0x00);
}

void write25BitsDbusData(){
  //0F 00 04 20 00 FD 07 40 00 16 18 80 00 04 20 00 01 08 40 00 02 10 80 00 00
  //0F 00 04 20 00 01 08 40 00 02 10 80 00 04 20 00 01 08 40 00 02 10 80 00 00
  
  Serial2.write(0x0F);
  Serial2.write((uint8_t) ( ((DBus_Output[0]&0x00FF)>>0) ) );//data1 0-7
  Serial2.write((uint8_t) ( ((DBus_Output[0]&0x0700)>>8) | ((DBus_Output[1]&0x001F)<<3) ) );//data1 8-10 data2 0-4 
  Serial2.write((uint8_t) ( ((DBus_Output[1]&0x07E0)>>5) | ((DBus_Output[2]&0x0003)<<6) ) );// data2 5-10 data3 0-1
  Serial2.write((uint8_t) ( ((DBus_Output[2]&0x03FC)>>2) ) );// data3 2-9
  Serial2.write((uint8_t) ( ((DBus_Output[2]&0x0400)>>10) | ((DBus_Output[3]&0x007F)<<1) ) );// data3 10 data 4 0-6
  Serial2.write((uint8_t) ( ((DBus_Output[3]&0x0780)>>7) | ((DBus_Output[4]&0x000F)<<4) ) );// data4 7-10 data 5 0-3
  Serial2.write((uint8_t) ( ((DBus_Output[4]&0x07F0)>>4) | ((DBus_Output[5]&0x0001)<<7) ) );// data4 7-10 data 5 0-3
  Serial2.write((uint8_t) ( ((DBus_Output[5]&0x01FE)>>1) ) );// data5 1-8
  Serial2.write((uint8_t) ( ((DBus_Output[5]&0x0600)>>9) | ((DBus_Output[6]&0x003F)<<2) ) );// data5 9-10 data 6 0-5
  Serial2.write((uint8_t) ( ((DBus_Output[6]&0x07C0)>>6) | ((0x00)<<4) ) );// data6 6-10 data 000     
  Serial2.write(0x80);
  Serial2.write(0x00);
  Serial2.write(0x04);
  Serial2.write(0x20);
  Serial2.write(0x00);
  Serial2.write(0x01);
  Serial2.write(0x08);
  Serial2.write(0x40);
  Serial2.write(0x00);
  Serial2.write(0x02);
  Serial2.write(0x10);
  Serial2.write(0x80);
  Serial2.write(0x00);
  Serial2.write(0x00);
}
