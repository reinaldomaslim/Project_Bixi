/*This node funtion(s):
  + Detect the red board and output its center
*/

//ROS libs
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Vector3.h>
#include <base_vision/board_colorConfig.h>
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
std::string published_topic;
std::string object_color;
bool debug;
//Image transport vars
cv_bridge::CvImagePtr cv_ptr;
//ROS var
std::vector<geometry_msgs::Vector3> object;
//OpenCV image processing method dependent vars 
std::vector<std::vector<cv::Point> > contours;
std::vector<cv::Point> out_contours, hull;
std::vector<cv::Vec4i> hierarchy;
std::vector<int> contour_index;
cv::Mat src, hsv, dst;
cv::Mat lower_hue_range, upper_hue_range;
cv::Mat str_el = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
cv::Scalar up_lim, low_lim, up_lim_wrap, low_lim_wrap;
cv::Point object_center;
cv::Rect rect;
int height, width;
int biggest_arrow_index;
int min_area = 50;
double area, biggest_arrow_area;
const double eps = 0.15;

//Functions
void reduce_noise(cv::Mat* dst)
{
  cv::morphologyEx(*dst, *dst, cv::MORPH_CLOSE, str_el);
  cv::morphologyEx(*dst, *dst, cv::MORPH_OPEN, str_el);
}

void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 0.4;
  int thickness = 1;
  int baseline = 0;

  cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
  cv::Rect r = cv::boundingRect(contour);

  cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
  cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
  cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void detect_board()
{
  biggest_arrow_area = 0; //initialization
  //Filter desired color
  if(object_color == "red")
  {//In case of red color
    cv::inRange(hsv, low_lim, up_lim, lower_hue_range);
    cv::inRange(hsv, low_lim_wrap, up_lim_wrap, upper_hue_range);
    cv::addWeighted(lower_hue_range,1.0,upper_hue_range,1.0,0.0,dst);
  }
  else cv::inRange(hsv, low_lim, up_lim, dst);
  //Reduce noise
  reduce_noise(&dst);
  //Finding shapes
  cv::findContours(dst.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  //Detect shape for each contour
  for(int i = 0; i < contours.size(); i++)
  {
    //Skip small objects
    area = cv::contourArea(contours[i]);
    if(area < min_area) continue;

    // Approximate contour with accuracy proportional
    // to the contour perimeter
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

    if(approx.size() == 4)
    {
      // Get the cosines of all corners
      std::vector<double> cos;
      for(int j = 2; j < 5; j++)
        cos.push_back(angle(approx[j%4], approx[j-2], approx[j-1]));

      // Sort ascending the cosine values
      std::sort(cos.begin(), cos.end());

      // Get the lowest and the highest cosine
      double mincos = cos.front();
      double maxcos = cos.back();

      // Use the degrees obtained above and the number of vertices
      // to determine the shape of the contour
      if (mincos >= -0.1 && maxcos <= 0.3)
      {
        rect = cv::boundingRect(contours[i]);
        object_center = (rect.tl() + rect.br() + cv::Point(1,1))*0.5;
        
        //Shows the numeric value on screen
        if(debug)
        {
          std::ostringstream ss;
          ss << object_center;
          std::string s(ss.str());
          setLabel(src, s, contours[i]);
        }
        cv::drawContours(src, contours, i, cv::Scalar(0,255,255), 2);
        geometry_msgs::Vector3 new_object;
        new_object.x = object_center.x;
        new_object.y = object_center.y;
        new_object.z = 0;
        object.push_back(new_object);
      }
    }
  }
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
  //Start the shape detection code
  cv::blur(src,src,Size(1,1));
  cv::cvtColor(src,hsv,COLOR_BGR2HSV);
  //Detect stuffs
  detect_board();
  //Show output on screen in debug mode
  if(debug) 
  {
    cv::imshow("src", src);
    cv::imshow("dst", dst);
  }
}

void dynamic_configCb(base_vision::board_colorConfig &config, uint32_t level) 
{
  min_area = config.min_area;
  if(object_color == "red")
  {
    low_lim = cv::Scalar(config.red_H_low1, config.red_S_low, config.red_V_low);
    up_lim = cv::Scalar(config.red_H_high1, config.red_S_high, config.red_V_high);
    low_lim_wrap = cv::Scalar(config.red_H_low2, config.red_S_low, config.red_V_low);
    up_lim_wrap = cv::Scalar(config.red_H_high2, config.red_S_high, config.red_V_high);
  }
  else if(object_color == "blue") 
  {
    low_lim = cv::Scalar(config.blue_H_low,config.blue_S_low,config.blue_V_low);
    up_lim = cv::Scalar(config.blue_H_high,config.blue_S_high,config.blue_V_high);
  }
  ROS_INFO("Reconfigure Requested.");
}

int main(int argc, char** argv)
{
  //Initiate node
  ros::init(argc, argv, "red_board");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  pnh.getParam("subscribed_image_topic", subscribed_image_topic);
  pnh.getParam("object_color", object_color);
  pnh.getParam("debug", debug);
  pnh.getParam("published_topic", published_topic);
  dynamic_reconfigure::Server<base_vision::board_colorConfig> server;
  dynamic_reconfigure::Server<base_vision::board_colorConfig>::CallbackType f;
  f = boost::bind(&dynamic_configCb, _1, _2);
  server.setCallback(f);
  
  //Initiate windows
  if(debug)
  {
    // cv::namedWindow("dst",WINDOW_AUTOSIZE);
    // cv::namedWindow("src",WINDOW_AUTOSIZE);
    cv::namedWindow("dst",WINDOW_NORMAL);
    cv::resizeWindow("dst",640,480);
    cv::moveWindow("dst", 0, 600);
    cv::namedWindow("src",WINDOW_NORMAL);
    cv::resizeWindow("src",640,480);
    cv::moveWindow("src", 0, 0);
    cv::startWindowThread();
  }
  //Start ROS subscriber...
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(subscribed_image_topic, 1, imageCb);
  //...and ROS publisher
  ros::Publisher pub = nh.advertise<geometry_msgs::Vector3>(published_topic, 1000);
  ros::Rate r(30);
  while (nh.ok())
  {
    //Publish every object detected
    for(vector<geometry_msgs::Vector3>::iterator it = object.begin(); it != object.end(); it++)
      pub.publish(*it);
    //Reinitialize the object counting vars
    object.clear();

    ros::spinOnce();
    r.sleep();
  }
  cv::destroyAllWindows();
  return 0;
}