#ifndef _ROS_hector_uav_msgs_RawMagnetic_h
#define _ROS_hector_uav_msgs_RawMagnetic_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"

namespace hector_uav_msgs
{

  class RawMagnetic : public ros::Msg
  {
    public:
      std_msgs::Header header;
      float channel[3];

    RawMagnetic():
      header(),
      channel()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      for( uint8_t i = 0; i < 3; i++){
      offset += serializeAvrFloat64(outbuffer + offset, this->channel[i]);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      for( uint8_t i = 0; i < 3; i++){
      offset += deserializeAvrFloat64(inbuffer + offset, &(this->channel[i]));
      }
     return offset;
    }

    const char * getType(){ return "hector_uav_msgs/RawMagnetic"; };
    const char * getMD5(){ return "babd510868ac7b486e2097c79e1384c9"; };

  };

}
#endif