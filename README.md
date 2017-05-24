<<<<<<< HEAD
Project_Bixi
============
Bixi robot for ICAR Robomasters 2017 competition

creator: Ren Ye (renye.china@gmail.com)

**This repository is INACTIVE**, please see [reinaldo](https://github.com/reinaldomaslim/Project_Bixi)'s repository for the latest


Changelog
---------
+ (2017-02-06) urdf-dev branch: description, gazebo, joint state, ros-control
+ (2017-01-30) vision-dev branch: opencv, fpv camera, arrow detection
+ (2017-01-29) create repository

TODO List
---------
(updated 2017-02-05)
+ arduino mega to drive mecanum wheels: needs to subscribe to geometry_msgs::Twist (`cmd_vel`) and advertise sensor_msgs::JointState (`encoder`)

+ vision to detect the arrow and cross, publish ROI, publish tracking info to `cmd_vel`


Required packages
-----------------
### ros ###
```bash
sudo apt get install ros-kinetic-ros-control ros-kinetic-ros-controllers
sudo apt get install ros-kinetic-gazebo-ros-control ros kinetic-gazebo-ros-pkgs
```
### opencv ###
```bash
todo
```
Reference and repositories
----------------------

+ husky:
    [repository](https://github.com/husky/husky.git)

+ neo_driver:
    [repository](https://github.com/neobotix/neo_driver/);
    [Mecanum4WKinematics.cpp](https://github.com/neobotix/neo_driver/blob/indigo_dev/neo_platformctrl_mecanum/common/src/Mecanum4WKinematics.cpp);
    [neo_platformctrl_node.cpp](https://github.com/neobotix/neo_driver/blob/indigo_dev/neo_platformctrl_mecanum/ros/src/neo_platformctrl_node.cpp)

+ rrbot:
    [repository](https://github.com/ros-simulation/gazebo_ros_demos.git)

+ mecanum wheel driver:
    [reference1/html](http://gmii.weebly.com/makeblock/4);
    [reference2/pdf](https://pdfs.semanticscholar.org/f971/36b372f36a588373142e17e8b68f6994227e.pdf);
    [reference3/pdf](http://download.makeblock.com/Mecanumbot/pdf%E9%BA%A6%E8%BD%AE%E4%B8%AD%E5%9E%8B%E5%BA%95%E7%9B%98%E6%90%AD%E5%BB%BA%E6%95%99%E7%A8%8B%20%EF%BC%88%E5%8F%8C%E6%9D%BF%20%EF%BC%89.pdf);
    [reference4/arduino](https://github.com/Makeblock-official/Mecanum-Wheel-Robot-Kit/blob/master/Mecanum_chassis_new/Mecanum_chassis_new.ino)
=======
# DJI_Arduino
ICRUA ROBOT rosconnection with arduino
>>>>>>> fcfd1b76a0f4ea13efdc11420a6f5b5e9534ba7a
