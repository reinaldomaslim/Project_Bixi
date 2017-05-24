#ifndef _ROS_mavros_ActuatorControl_h
#define _ROS_mavros_ActuatorControl_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"

namespace mavros
{

  class ActuatorControl : public ros::Msg
  {
    public:
      std_msgs::Header header;
      uint8_t group_mix;
      float controls[8];

    ActuatorControl():
      header(),
      group_mix(0),
      controls()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      *(outbuffer + offset + 0) = (this->group_mix >> (8 * 0)) & 0xFF;
      offset += sizeof(this->group_mix);
      for( uint8_t i = 0; i < 8; i++){
      union {
        float real;
        uint32_t base;
      } u_controlsi;
      u_controlsi.real = this->controls[i];
      *(outbuffer + offset + 0) = (u_controlsi.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_controlsi.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_controlsi.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_controlsi.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->controls[i]);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      this->group_mix =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->group_mix);
      for( uint8_t i = 0; i < 8; i++){
      union {
        float real;
        uint32_t base;
      } u_controlsi;
      u_controlsi.base = 0;
      u_controlsi.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_controlsi.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_controlsi.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_controlsi.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->controls[i] = u_controlsi.real;
      offset += sizeof(this->controls[i]);
      }
     return offset;
    }

    const char * getType(){ return "mavros/ActuatorControl"; };
    const char * getMD5(){ return "ba56d1663d64ac435b80a3508d6ce1d2"; };

  };

}
#endif