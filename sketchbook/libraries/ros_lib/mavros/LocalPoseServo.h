#ifndef _ROS_mavros_LocalPoseServo_h
#define _ROS_mavros_LocalPoseServo_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace mavros
{

  class LocalPoseServo : public ros::Msg
  {
    public:
      float servo1;
      float servo2;

    LocalPoseServo():
      servo1(0),
      servo2(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        float real;
        uint32_t base;
      } u_servo1;
      u_servo1.real = this->servo1;
      *(outbuffer + offset + 0) = (u_servo1.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_servo1.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_servo1.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_servo1.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->servo1);
      union {
        float real;
        uint32_t base;
      } u_servo2;
      u_servo2.real = this->servo2;
      *(outbuffer + offset + 0) = (u_servo2.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_servo2.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_servo2.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_servo2.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->servo2);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        float real;
        uint32_t base;
      } u_servo1;
      u_servo1.base = 0;
      u_servo1.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_servo1.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_servo1.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_servo1.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->servo1 = u_servo1.real;
      offset += sizeof(this->servo1);
      union {
        float real;
        uint32_t base;
      } u_servo2;
      u_servo2.base = 0;
      u_servo2.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_servo2.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_servo2.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_servo2.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->servo2 = u_servo2.real;
      offset += sizeof(this->servo2);
     return offset;
    }

    const char * getType(){ return "mavros/LocalPoseServo"; };
    const char * getMD5(){ return "57c9bc16c2ecc3edfcfb0fb8e47e5b1f"; };

  };

}
#endif