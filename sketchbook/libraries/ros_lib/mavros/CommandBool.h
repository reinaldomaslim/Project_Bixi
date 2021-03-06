#ifndef _ROS_SERVICE_CommandBool_h
#define _ROS_SERVICE_CommandBool_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace mavros
{

static const char COMMANDBOOL[] = "mavros/CommandBool";

  class CommandBoolRequest : public ros::Msg
  {
    public:
      bool value;

    CommandBoolRequest():
      value(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_value;
      u_value.real = this->value;
      *(outbuffer + offset + 0) = (u_value.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->value);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_value;
      u_value.base = 0;
      u_value.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->value = u_value.real;
      offset += sizeof(this->value);
     return offset;
    }

    const char * getType(){ return COMMANDBOOL; };
    const char * getMD5(){ return "e431d687bf4b2c65fbd94b12ae0cb5d9"; };

  };

  class CommandBoolResponse : public ros::Msg
  {
    public:
      bool success;
      uint8_t result;

    CommandBoolResponse():
      success(0),
      result(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.real = this->success;
      *(outbuffer + offset + 0) = (u_success.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->success);
      *(outbuffer + offset + 0) = (this->result >> (8 * 0)) & 0xFF;
      offset += sizeof(this->result);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_success;
      u_success.base = 0;
      u_success.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->success = u_success.real;
      offset += sizeof(this->success);
      this->result =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->result);
     return offset;
    }

    const char * getType(){ return COMMANDBOOL; };
    const char * getMD5(){ return "1cd894375e4e3d2861d2222772894fdb"; };

  };

  class CommandBool {
    public:
    typedef CommandBoolRequest Request;
    typedef CommandBoolResponse Response;
  };

}
#endif
