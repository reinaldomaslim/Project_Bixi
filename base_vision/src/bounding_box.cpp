/*This node funtion(s):
	+ Outputs all possible bounding boxes of armor plates
  + Output type = ROI
*/

//ROS libs
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <base_vision/armor_colorConfig.h>
//OpenCV libs
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//C++ standard libs
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
//Namespaces
using namespace ros;
using namespace cv;
using namespace std;
//ROS params
std::string subscribed_image_topic;
std::string armor_color;
std::string published_topic;
bool debug;
//Image transport vars
cv_bridge::CvImagePtr cv_ptr;
//ROS var
vector<sensor_msgs::RegionOfInterest> object;
//OpenCV image processing method dependent vars 
std::vector<std::vector<cv::Point> > contours;
std::vector<cv::Vec4i> hierarchy;
std::vector<int> contour_index;
cv::Mat src, croppedRef, cropped, hsv, dst, dst_bgr;
cv::Scalar up_lim, low_lim, up_lim_wrap, low_lim_wrap;
cv::Mat lower_hue_range, upper_hue_range;
cv::Mat str_el = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
cv::Rect rect;
cv::RotatedRect mr;
int height, width;
int min_area = 50;
double area, r_area, mr_area;
const double eps = 0.15;

//Functions
void reduce_noise(cv::Mat* dst)
{
  cv::morphologyEx(*dst, *dst, cv::MORPH_CLOSE, str_el);
  cv::morphologyEx(*dst, *dst, cv::MORPH_OPEN, str_el);
}

void armor_found(int a, int b, float LED_length)
{
  cv::Rect recta = cv::boundingRect(contours[a]);
  cv::Rect rectb = cv::boundingRect(contours[b]);
  sensor_msgs::RegionOfInterest obj;
  if(recta.tl().x > rectb.tl().x)
  {
    cv::Rect temp = recta;
    recta = rectb;
    rectb = temp;
  }
  float x_offset = (recta.tl().x);
    obj.x_offset = x_offset;
  float y_offset = ((recta.tl().y + rectb.tl().y)/2.0 - LED_length + height/4); //note that the cropping makes change to the coordinate, which explains the height/4 component
    obj.y_offset = y_offset;
  float armor_height = 3*LED_length;
    obj.height = armor_height;
  float armor_width = fabs(recta.tl().x - rectb.br().x);
    obj.width = armor_width;
    obj.do_rectify = true;
  object.push_back(obj);  //Push the object to the vector
  if(debug) cv::rectangle(src, cv::Point(x_offset, y_offset), cv::Point(x_offset + armor_width, y_offset + armor_height), cv::Scalar(255,0,255), 2, 8, 0);
  return;
}

void detect_armor()
{
  //Filter desired color
  if(armor_color == "red")
  {//In case of red color
    cv::inRange(hsv, low_lim, up_lim, lower_hue_range);
    cv::inRange(hsv, low_lim_wrap, up_lim_wrap, upper_hue_range);
    cv::addWeighted(lower_hue_range,1.0,upper_hue_range,1.0,0.0,dst);
  }
  else cv::inRange(hsv, low_lim, up_lim, dst);
  cv::imshow("bin", dst);
  //Reduce noise
  reduce_noise(&dst);
  //Finding shapes
  cv::findContours(dst.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  //Detect shape for each contour
  for(int i = 0; i < contours.size(); i++)
  {
    rect = cv::boundingRect(contours[i]);
    area = cv::contourArea(contours[i]);
    //Delete the contours that are too small or not seem to be the LED on the armor
    if(area < min_area)
    {
      if(debug) cv::drawContours(dst, contours, i, Scalar(0,0,0),-1);
      continue;
    }
    if(rect.height < rect.width*2)
    {
      if(debug) cv::drawContours(dst, contours, i, Scalar(0,0,0),-1);
      continue;
    }
    mr = cv::minAreaRect(contours[i]);
    mr_area = (mr.size).height*(mr.size).width;
    if(area/mr_area < 0.5)
    {
      if(debug) cv::drawContours(dst, contours, i, Scalar(0,0,0),-1);
      continue;
    }
    //If contour is good, register the contour for further processing
    contour_index.push_back(i);
  }
  if(contour_index.empty()) return;
  //Final process
  cv::cvtColor(dst,dst_bgr,COLOR_GRAY2BGR);
  for(int i = 0; i < contour_index.size()-1; i++)
    for(int j = i+1; j < contour_index.size(); j++)
    {
      if(i==j) continue;
      cv::RotatedRect rect1 = cv::minAreaRect(contours[contour_index[i]]);
      cv::RotatedRect rect2 = cv::minAreaRect(contours[contour_index[j]]);
      //Check distance between 2 LEDs
      float LED_length = max(max(rect1.size.height, rect1.size.width), max(rect2.size.height, rect2.size.width));
      armor_found(contour_index[i], contour_index[j], LED_length);
      ROS_INFO("Found %d %d %f", i, j, LED_length);
    }
  contour_index.clear();
  return;
}

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  //Get the image in OpenCV format
  src = cv_ptr->image;
  if(src.empty())
  {
    if(debug) ROS_INFO("Empty input. Looping...");
    return;
  }
  width = src.cols;
  height = src.rows;
  //Crop image, note that this doesnt copy the data
  croppedRef = src(cv::Rect(0, height/4, width, 3*height/4));
  croppedRef.copyTo(cropped);
  //Start the shape detection code
  cv::blur(cropped,cropped,Size(1,1));
  cv::cvtColor(cropped,hsv,COLOR_BGR2HSV);
  //Detect stuffs
  detect_armor();
  //Show output on screen in debug mode
  if(debug) 
  {
    cv::imshow("src", src);
    cv::imshow("dst", dst);
  }
}

void dynamic_configCb(base_vision::armor_colorConfig &config, uint32_t level) 
{
  min_area = config.min_area;
  //Process appropriate parameter for armor color
  if(armor_color == "blue") 
  {
    low_lim = cv::Scalar(config.blue_H_low,config.blue_S_low,config.blue_V_low);
    up_lim = cv::Scalar(config.blue_H_high,config.blue_S_high,config.blue_V_high);
  }
  else if(armor_color == "red")
  {
    low_lim = cv::Scalar(config.red_H_low1, config.red_S_low, config.red_V_low);
    up_lim = cv::Scalar(config.red_H_high1, config.red_S_high, config.red_V_high);
    low_lim_wrap = cv::Scalar(config.red_H_low2, config.red_S_low, config.red_V_low);
    up_lim_wrap = cv::Scalar(config.red_H_high2, config.red_S_high, config.red_V_high);
  }
  ROS_INFO("Reconfigure Requested.");
}

int main(int argc, char** argv)
{
  //Initiate node
  ros::init(argc, argv, "armor_detection");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  pnh.getParam("subscribed_image_topic", subscribed_image_topic);
  pnh.getParam("armor_color", armor_color);
  pnh.getParam("debug", debug);
  pnh.getParam("published_topic", published_topic);
  dynamic_reconfigure::Server<base_vision::armor_colorConfig> server;
  dynamic_reconfigure::Server<base_vision::armor_colorConfig>::CallbackType f;
  f = boost::bind(&dynamic_configCb, _1, _2);
  server.setCallback(f);
  
  //Initiate windows
  if(debug)
  {
   /* cv::namedWindow("color",WINDOW_AUTOSIZE);
    cv::namedWindow("src",WINDOW_AUTOSIZE);*/
    cv::namedWindow("bin",WINDOW_NORMAL);
    cv::resizeWindow("bin",640,480);
    cv::namedWindow("dst",WINDOW_NORMAL);
    cv::resizeWindow("dst",640,480);
    cv::namedWindow("src",WINDOW_NORMAL);
    cv::resizeWindow("src",640,480);
    cv::startWindowThread();
  }
  //Start ROS subscriber...
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(subscribed_image_topic, 1, imageCb);
  //...and ROS publisher
  ros::Publisher pub = nh.advertise<sensor_msgs::RegionOfInterest>(published_topic, 1000);
  ros::Rate r(30);
  while (nh.ok())
  {
  	//Publish every object detected
    for(vector<sensor_msgs::RegionOfInterest>::iterator it = object.begin(); it != object.end(); it++)
      pub.publish(*it);
    //Reinitialize the object counting vars
    object.clear();

    ros::spinOnce();
    r.sleep();
  }
  cv::destroyAllWindows();
  return 0;
}