#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>

std::string video_file;
cv::Mat image;  
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "video_publisher");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  pnh.getParam("video_file", video_file);
  cv::VideoCapture cap(video_file);

  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("image_rect_color", 1);
  cv::waitKey(30);
  ROS_INFO("Press p to play/pause.");
  ros::Rate loop_rate(10);
  bool play = true;
  while (nh.ok()) 
  {
    if(play == true)
    {
      cap >> image;
      if(image.empty())
        cap.set(2,0);
    }
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}