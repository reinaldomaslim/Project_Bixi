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
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
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
std::string subscribed_laser_topic;
std::string published_topic;
bool debug;
//Image transport vars
cv_bridge::CvImagePtr cv_ptr;
//ROS var
vector<geometry_msgs::Point> destination_position;
//OpenCV image processing method dependent vars 
std::vector<std::vector<cv::Point> > contours;
std::vector<cv::Point> out_contours, hull;
std::vector<cv::Vec4i> hierarchy;
std::vector<int> contour_index;
cv::Mat src, hsv, dst, gray, dst2, detected_edges;
cv::Mat lower_hue_range, upper_hue_range;
cv::Mat str_el = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
cv::Scalar up_lim1, low_lim1, up_lim2, low_lim2;
cv::Point arrow_direction, biggest_arrow_direction, arrow_center;
cv::Rect rect;
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

//Classes
class Box
{
public:
  geometry_msgs::Pose box_pose;
  ros::Time box_time;
  Box(geometry_msgs::Pose new_box_pose, ros::Time new_box_time)
    : box_pose(new_box_pose), box_time(new_box_time)
  {}
  double box_angle = -std::tan(box_pose.position.y/box_pose.position.x);
  int occurance = 0;
};

class Shape
{
public:
  int type;       //type = 0 for circles; type = 1 for arrows
  int direction;  //direction = 0 for left; direction = 1 for right; for circles, this parameter has no meaning
  cv::Point center;
  Shape(int new_shape_type, cv::Point new_shape_center, int new_shape_direction)
    : type(new_shape_type), center(new_shape_center), direction(new_shape_direction)
  {}
};

std::vector<Box> detected_box_vector;
std::vector<Shape> detected_shape_vector;
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

	  //Find the arrow's center
    cv::Point2f vertices[4];
    mr.points(vertices);
    arrow_center = (vertices[0] + vertices[2])*0.5;

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

    //Check the direction of the arrow
    arrow_direction = cv::Point(0,0); //initialization
    for(int j = 0; j < out_contours.size(); j++) 
    	arrow_direction += (out_contours[j] - arrow_center);
    //Save all detected arrows into a vector to combine with data from laser:
    if(arrow_direction.x > 0) 
    {
      Shape temp_shape_holder(1,arrow_center,1);
      detected_shape_vector.push_back(temp_shape_holder);
    }
    else
    {
      Shape temp_shape_holder(1,arrow_center,0);
      detected_shape_vector.push_back(temp_shape_holder);
    }
    
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
  params.filterByArea = true;         // filter my blobs by area of blob
  params.minThreshold = 0;
  params.maxThreshold = 100;
  params.minArea = 300.0;              // min pixels squared
  // params.maxArea = 1000000.0;             // max pixels squared
  params.filterByCircularity = true;
  params.minCircularity = 0.7;
  SimpleBlobDetector myBlobDetector(params);
  std::vector<cv::KeyPoint> myBlobs;
  myBlobDetector.detect(gray, myBlobs);

  cv::Mat blobImg;
  std::vector<cv::Point> detected_circles;
  for(std::vector<cv::KeyPoint>::iterator blobIterator = myBlobs.begin(); blobIterator != myBlobs.end(); blobIterator++)
  {
    cv::Point center(blobIterator->pt.x, blobIterator->pt.y);
    int radius = cvRound(blobIterator->size);
    if(debug) cv::circle(src, center, radius, Scalar(0,0,255), 3, 8, 0);
    // std::cout << "size of blob is: " << blobIterator->size << std::endl;
    // std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;

    //Save all detected circles into a vector to combine with data from laser:
    Shape temp_shape_holder(0,center,0);
    detected_shape_vector.push_back(temp_shape_holder);
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
  detected_shape_vector.clear();
  detect_arrow();
  detect_circle();
  //Visualization on screen
  if(debug) 
  {
    cv::imshow("src", src);
    cv::imshow("black", dst);
    // cv::imshow("blue", dst2);
  }

  // ROS_INFO("detected_shape_vector[data]\n");
  // for(std::vector<Shape>::iterator it = detected_shape_vector.begin(); it != detected_shape_vector.end(); it++)
  // {
  //   cout << "Object type = " << it->type << endl;
  //   if(it->type != 0) cout << "Object direction = " << it->direction << endl;
  //   cout << "Object center = " << it->center << endl << endl;
  // }

  //Do the matching
  if(detected_shape_vector.empty() || detected_box_vector.empty()) return; //if either vector is empty: exit
  for(vector<Box>::iterator it = detected_box_vector.begin(); it != detected_box_vector.end(); ++it)
  {
    std::vector<Shape> matched_shape;
    for(vector<Shape>::iterator it2 = detected_shape_vector.begin(); it2 != detected_shape_vector.end(); it2++)
    {
      double shape_angle_pos = (2*((double)it2->center.x)/width-1)*39*180/CV_PI;        //horizontal view of camera is 78 degrees
      if(fabs(shape_angle_pos - it->box_angle) < max_angle_difference)
        matched_shape.push_back(*it2);    //Record all shapes that match the angle with the box
    }
    if(matched_shape.empty()) continue; //no matches
    Shape chosen_shape = *matched_shape.begin();
    if(matched_shape.size() > 1) //If there are many shapes that matched: choose the bottom-most one in the image (thus, the one with the highest y-value)
    {
      for(vector<Shape>::iterator it2 = matched_shape.begin()+1; it2 != matched_shape.end(); it2++)
        if(chosen_shape.center.y < it2->center.y) chosen_shape = *it2;
    }

    // The pair is: *it (for box) and chosen_shape (for shape)
    // Calculate the destination location for navigation node
    if(chosen_shape.type == 0) //Circle
    {
      geometry_msgs::Point temp_destination_holder;
      //Transform quaternion data into angle data (yaw)
      tf::Quaternion q(it->box_pose.orientation.x, it->box_pose.orientation.y, it->box_pose.orientation.z, it->box_pose.orientation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      //Calculate destination for navigation node from data
      temp_destination_holder.x = it->box_pose.position.x - offset_distance*sin(yaw); //x' = x - d*sin(yaw)
      temp_destination_holder.y = it->box_pose.position.y - offset_distance*cos(yaw); //y' = y - d*cos(yaw)
      destination_position.push_back(temp_destination_holder);
    }else if((chosen_shape.type == 1) && (chosen_shape.direction == 0)) //Arrow with left direction
    {
      geometry_msgs::Point temp_destination_holder;
      //Transform quaternion data into angle data (yaw)
      tf::Quaternion q(it->box_pose.orientation.x, it->box_pose.orientation.y, it->box_pose.orientation.z, it->box_pose.orientation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      //Calculate destination for navigation node from data
      temp_destination_holder.x = it->box_pose.position.x + (box_size/2)*(-cos(yaw)+ sin(yaw)) - offset_distance*cos(yaw); //x' = x - a/2*cos(yaw) + a/2*sin(yaw) - d*cos(yaw)
      temp_destination_holder.y = it->box_pose.position.y + (box_size/2)*(sin(yaw) + cos(yaw)) + offset_distance*sin(yaw); //y' = y + a/2*sin(yaw) + a/2*cos(yaw) + d*sin(yaw)
      destination_position.push_back(temp_destination_holder);
    }else if((chosen_shape.type == 1) && (chosen_shape.direction == 1)) //Arrow with right direction
    {
      geometry_msgs::Point temp_destination_holder;
      //Transform quaternion data into angle data (yaw)
      tf::Quaternion q(it->box_pose.orientation.x, it->box_pose.orientation.y, it->box_pose.orientation.z, it->box_pose.orientation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      //Calculate destination for navigation node from data
      temp_destination_holder.x = it->box_pose.position.x + (box_size/2)*(cos(yaw) + sin(yaw)) + offset_distance*cos(yaw); //x' = x + a/2*cos(yaw) + a/2*sin(yaw) + d*cos(yaw)
      temp_destination_holder.y = it->box_pose.position.y + (box_size/2)*(-sin(yaw)+ cos(yaw)) - offset_distance*sin(yaw); //y' = y - a/2*sin(yaw) + a/2*cos(yaw) - d*sin(yaw)
      destination_position.push_back(temp_destination_holder);
    }
  }
}

void laserCb(const geometry_msgs::PoseStampedConstPtr& msg)
{
  ros::Time current_time = ros::Time::now();
  Box new_box(msg->pose, current_time);
  detected_box_vector.push_back(new_box);
  while(!detected_box_vector.empty()) 
  {
    ros::Duration box_age = current_time - detected_box_vector[0].box_time;
    if(box_age.toSec() > 1) detected_box_vector.erase(detected_box_vector.begin());
    else break;
  }
  std::vector<Box>::iterator it = detected_box_vector.begin();
  while(it < (detected_box_vector.end()-1))
  {
    geometry_msgs::Point current_box_pos = it->box_pose.position;
    geometry_msgs::Point next_box_pos = (it+1)->box_pose.position;
    double distance_currentbox_nextbox = sqrt(pow((current_box_pos.x - next_box_pos.x),2) + pow((current_box_pos.y - next_box_pos.y),2) + pow((current_box_pos.z - next_box_pos.z),2));
    if(distance_currentbox_nextbox < 0.5) 
    {
      detected_box_vector.erase(it+1);
      it->occurance++;
    }
    else it++;
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
  pnh.getParam("subscribed_laser_topic", subscribed_laser_topic);
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
    // cv::namedWindow("blue",WINDOW_NORMAL);
    // cv::resizeWindow("blue",640,480);
    // cv::moveWindow("blue", 500, 600);
    cv::setMouseCallback("src", onMouse, 0);
    cv::startWindowThread();
  }
  //Start ROS subscriber...
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber image_sub = it.subscribe(subscribed_image_topic, 1, imageCb);
  ros::Subscriber laser_sub = nh.subscribe(subscribed_laser_topic, 1, laserCb);
  //...and ROS publisher
  ros::Publisher pub = nh.advertise<geometry_msgs::Point>(published_topic, 1000);
  ros::Rate r(30);
  while (nh.ok())
  {
  	//Publish every object detected
    for(vector<geometry_msgs::Point>::iterator it = destination_position.begin(); it != destination_position.end(); it++)
      pub.publish(*it);
    //Reinitialize the object counting vars
    destination_position.clear();

    ros::spinOnce();
    r.sleep();
  }
  cv::destroyAllWindows();
  return 0;
}