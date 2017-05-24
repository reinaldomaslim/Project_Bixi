/*This node funtion(s):
	+ Detects arrows and outputs the biggest arrow's direction
*/

//ROS libs
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <tf/transform_datatypes.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <base_vision/building_blocksConfig.h>
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
#include <math.h>
//Namespaces
using namespace ros;
using namespace cv;
using namespace std;
//ROS params
std::string subscribed_image_topic;
std::string published_topic;
bool debug;
//Image transport vars
cv_bridge::CvImagePtr cv_ptr;
//ROS var
vector<geometry_msgs::Vector3> output_objects;
//OpenCV image processing method dependent vars 
std::vector<std::vector<cv::Point> > contours;
std::vector<cv::Point> out_contours, hull;
std::vector<cv::Vec4i> hierarchy;
std::vector<int> contour_index;
cv::Mat src, hsv, dst, gray, dst2;
cv::Mat lower_hue_range, upper_hue_range;
cv::Mat str_el = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
cv::Scalar up_lim1, low_lim1, up_lim2, low_lim2;
cv::Point arrow_direction, biggest_arrow_direction, arrow_center;
cv::RotatedRect mr;
int height, width;
int biggest_arrow_index;
int min_area = 500;
int cannyThreshold, accumulatorThreshold;
double area, mr_area, hull_area;
double max_angle_difference = 10*180/CV_PI;
double box_size = 0.2; //0.2m
double offset_distance = 0.2; //The robot will go to the point 0.2m in front of the hole of the box
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

void detect_arrow()
{
  //Filter desired color
  cv::inRange(hsv, low_lim1, up_lim1, dst);
  //Reduce noise
  reduce_noise(&dst);
  //Finding shapes
  cv::findContours(dst.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  //Detect shape for each contour
  for(int i = 0; i < contours.size(); i++)
  {
  	//Skip small objects
  	area = cv::contourArea(contours[i]);
    if(area < min_area) continue;

    //Calculate contour areas
	  mr = cv::minAreaRect(contours[i]);
    mr_area = (mr.size).height*(mr.size).width;
    cv::convexHull(contours[i], hull, 0, 1);
    hull_area = contourArea(hull);

		//Check if the number of convex corners is 5
	  cv::approxPolyDP(cv::Mat(hull), out_contours, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
	  if(out_contours.size() != 5) 
	  	continue;

		//Check if the number of corners is 7
		cv::approxPolyDP(cv::Mat(contours[i]), out_contours, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
		if(out_contours.size() != 7)
			continue;

    //Check if the dominant color inside the contour is blue
    int total_points = 0;
    int blue_points = 0;
    for(int image_col = 0; image_col <= width; image_col++)
      for(int image_row = 0; image_row <= height; image_row++)
        if(cv::pointPolygonTest(contours[i], cv::Point(image_col, image_row), false) == 1) //Check if points in inside the contour
        {
          total_points++;
          cv::Vec3b pixel = hsv.at<Vec3b>(cv::Point(image_col,image_row));
          if(fabs(pixel.val[0] - 108) < 10) blue_points++;
        }
    if(blue_points*2.5 < total_points) continue; //skip if not blue color dominant

    //Check if the area ratios are within allowed range
    if((fabs(area/mr_area - 0.6) > 0.07) && (fabs(hull_area/mr_area - 0.78) > 0.07))
      continue;

    //Code from here is only run if the shape is confirmed arrow
    //Find the arrow's center
    cv::Point2f vertices[4];
    mr.points(vertices);
    arrow_center = (vertices[0] + vertices[2])*0.5;
    //Check the direction of the arrow
    arrow_direction = cv::Point(0,0); //initialization
    for(int j = 0; j < out_contours.size(); j++) 
    	arrow_direction += (out_contours[j] - arrow_center);
    //Save all detected arrows into the output vector to publish
    geometry_msgs::Vector3 temp_holder;
    if(arrow_direction.x < 0) //Direction == left
      temp_holder.x = 0;
    else temp_holder.x = 1; //Direction == right
    temp_holder.y = 0;
    temp_holder.z = (2*((double)arrow_center.x)/width-1)*39*180/CV_PI;        //horizontal view of camera is 78 degrees, all angles are in radians
    output_objects.push_back(temp_holder);

		//Visualization on detected arrows
    if(debug)
    {
      cv::drawContours(src, contours, i, cv::Scalar(0,255,255), 2);
  		std::ostringstream ss;
  		if(arrow_direction.x > 0) ss << "R";
  		  else ss << "L";
      std::string s(ss.str());
      setLabel(src, s, contours[i]);
    }
  }
  return;
}

void detect_circle()
{
  //Convert src to grayscale
  cv::cvtColor(src, gray, COLOR_BGR2GRAY);
  cv::SimpleBlobDetector::Params params; 
  params.minDistBetweenBlobs = 50.0;  // minimum 10 pixels between blobs
  params.filterByColor = true;
  params.minThreshold = 0;
  params.maxThreshold = 50;
  params.thresholdStep = 5;
  params.filterByArea = true;         // filter my blobs by area of blob
  params.minArea = 200.0;              // min pixels squared
  // params.maxArea = 1000000.0;             // max pixels squared
  params.filterByCircularity = true;
  params.minCircularity = 0.5;
  SimpleBlobDetector myBlobDetector(params);
  std::vector<cv::KeyPoint> myBlobs;
  myBlobDetector.detect(gray, myBlobs);

  for(std::vector<cv::KeyPoint>::iterator blobIterator = myBlobs.begin(); blobIterator != myBlobs.end(); blobIterator++)
  {
    cv::Point center(blobIterator->pt.x, blobIterator->pt.y);
    int radius = cvRound(blobIterator->size);
    if(debug) cv::circle(src, center, radius, Scalar(0,0,255), 3, 8, 0);
    // std::cout << "size of blob is: " << blobIterator->size << std::endl;
    // std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;

    //Save all detected circles into the output vector to publish
    geometry_msgs::Vector3 temp_holder;
    temp_holder.x = 2;
    temp_holder.y = 0;     
    temp_holder.z = (2*((double)center.x-1)/width-1)*39*180/CV_PI;        //all angles are in radians
    output_objects.push_back(temp_holder);
  } 
  if(debug) cv::drawKeypoints(src, myBlobs, src);
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
  cv::blur(src,src,Size(3,3));
  cv::cvtColor(src,hsv,COLOR_BGR2HSV);
  //Detect stuffs and show output on screen (in debug mode)
  detect_arrow();
  detect_circle();
  //Visualization on screen
  if(debug) 
  {
    cv::imshow("src", src);
    cv::imshow("black", gray);
    // cv::imshow("blue", dst2);
  }
}

void dynamic_configCb(base_vision::building_blocksConfig &config, uint32_t level) 
{
  min_area = config.min_area;
  low_lim1 = cv::Scalar(config.black_H_low,config.black_S_low,config.black_V_low);
  up_lim1 = cv::Scalar(config.black_H_high,config.black_S_high,config.black_V_high);
  low_lim2 = cv::Scalar(config.blue_H_low,config.blue_S_low,config.blue_V_low);
  up_lim2 = cv::Scalar(config.blue_H_high,config.blue_S_high,config.blue_V_high);
  cannyThreshold = config.cannyThreshold;
  accumulatorThreshold = config.accumulatorThreshold;
  ROS_INFO("Reconfigure Requested.");
}

static void onMouse(int event, int x, int y, int, void*)
{
  if(event == EVENT_LBUTTONDOWN)
  { 
    Vec3b pixel = hsv.at<Vec3b>(cv::Point(x,y));
    std::cout << "\tAt point [" << x << "," << y << "]: (" << (float)pixel.val[0] << ", " << (float)pixel.val[1] << ", " << (float)pixel.val[2] << ")\n";
  }
}

int main(int argc, char** argv)
{
  //Initiate node
  ros::init(argc, argv, "arrow");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  pnh.getParam("subscribed_image_topic", subscribed_image_topic);
  pnh.getParam("debug", debug);
  pnh.getParam("published_topic", published_topic);
  //Dynamic reconfigure option
  dynamic_reconfigure::Server<base_vision::building_blocksConfig> server;
  dynamic_reconfigure::Server<base_vision::building_blocksConfig>::CallbackType f;
  f = boost::bind(&dynamic_configCb, _1, _2);
  server.setCallback(f);
  
  //Initiate windows
  if(debug)
  {
   /* cv::namedWindow("color",WINDOW_AUTOSIZE);
    cv::namedWindow("src",WINDOW_AUTOSIZE);*/
    cv::namedWindow("src",WINDOW_NORMAL);
    cv::resizeWindow("src",640,480);
    cv::moveWindow("src", 0, 0);
    cv::namedWindow("black",WINDOW_NORMAL);
    cv::resizeWindow("black",640,480);
    cv::moveWindow("black", 0, 600);
    cv::setMouseCallback("src", onMouse, 0);
    cv::startWindowThread();
  }
  //Start ROS subscriber...
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber image_sub = it.subscribe(subscribed_image_topic, 1, imageCb);
  //...and ROS publisher
  ros::Publisher pub = nh.advertise<geometry_msgs::Vector3>(published_topic, 1000);
  ros::Rate r(30);
  while (nh.ok())
  {
  	//Publish every object detected
    for(vector<geometry_msgs::Vector3>::iterator it = output_objects.begin(); it != output_objects.end(); it++)
      pub.publish(*it);
    //Reinitialize the object counting vars
    output_objects.clear();

    ros::spinOnce();
    r.sleep();
  }
  cv::destroyAllWindows();
  return 0;
}