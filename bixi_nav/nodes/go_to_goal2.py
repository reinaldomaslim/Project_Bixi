#!/usr/bin/env python  
import roslib
import rospy
from nav_msgs.msg import Odometry

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped
from sensor_msgs.msg import Joy
from tf.transformations import quaternion_from_euler, euler_from_quaternion



import math

class GoToGoal(object):
    x0, y0, yaw0= 0, 0, 0
    goal_des=[0, 0, 0]
    initialize=True

    ## PID constants
    del_T = 100.0   # time step in ms
    p_ang = 600.0
    d_ang = 300.0
    p_lin = 600.0
    d_lin = 300.0
    lin_vel_thres = 200.0 # max 660
    ang_vel_thres = 200.0 # max 660
    bias = 1024.0
    pre_ang_error = 0.0
    pre_x_error = 0.0
    pre_y_error = 0.0

    def __init__(self, nodename):
        heading_threshold=5*math.pi/180

        rospy.init_node('go_to_goal')

        rospy.Subscriber("/odometry", Odometry, self.odom_callback, queue_size = 50)
        rospy.sleep(1)
        rospy.Subscriber("/target_goal", PoseStamped, self.goal_callback , queue_size=10)

        self.cmd_vel_pub=rospy.Publisher("/vel_cmd", Joy, queue_size=10)

        r = rospy.Rate(1/self.del_T*1000)

        while not rospy.is_shutdown():

            #if direction not similar, rotate
            if abs(self.yaw0-self.goal_des[2])>heading_threshold:
                self.rotate(self.goal_des[2])
            else:
                #else translate to goal    
                self.translate(self.goal_des[0], self.goal_des[1])


            r.sleep()

    def goal_callback(self, msg):

        #store target goal as private variable
        _, _, yaw_des = euler_from_quaternion((msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))
        self.goal_des=[msg.pose.position.x, msg.pose.position.y, yaw_des]


    def rotate(self, angle):
        msg=Joy()
        
        ang_error=math.atan2(math.sin(angle-self.yaw0), math.cos(angle-self.yaw0))
        derivative = (ang_error - self.pre_ang_error) / self.del_T
        angular_vel = (self.p_ang * ang_error) + (self.d_ang * derivative)

        # if abs(ang_error)<math.pi:
        #     msg.angular.z=1024+angular_vel
        # else:
        #     msg.angular.z=1024-angular_vel

        if angular_vel > self.ang_vel_thres:
            angular_vel = self.ang_vel_thres
        elif angular_vel < -self.ang_vel_thres:
            angular_vel = -self.ang_vel_thres


        theta = self.bias - angular_vel

        msg.buttons = [theta, self.bias, self.bias]
        self.cmd_vel_pub.publish(msg)
        
        self.pre_ang_error = ang_error

    def translate(self, x_target, y_target):
        msg=Joy()
        # vel=200 #must be small to avoid jerking, and secondly to avoid switching surface
        # distance_threshold=0.1

        x_error=(x_target-self.x0)*math.cos(self.yaw0)+(y_target-self.y0)*math.sin(self.yaw0)
        y_error=-(x_target-self.x0)*math.sin(self.yaw0)+(y_target-self.y0)*math.cos(self.yaw0)


        x_derivative = (x_error - self.pre_x_error) / self.del_T
        y_derivative = (y_error - self.pre_y_error) / self.del_T
        
        x_linear_vel = (self.p_lin * x_error) + (self.d_lin * x_derivative)
        
        if x_linear_vel > self.lin_vel_thres:
            x_linear_vel = self.lin_vel_thres
        elif x_linear_vel < -self.lin_vel_thres:
            x_linear_vel = -self.lin_vel_thres
        x = self.bias + x_linear_vel

        y_linear_vel = (self.p_lin * y_error) + (self.d_lin * y_derivative)
        if y_linear_vel > self.lin_vel_thres:
            y_linear_vel = self.lin_vel_thres
        elif y_linear_vel < -self.lin_vel_thres:
            y_linear_vel = -self.lin_vel_thres
        y = self.bias + y_linear_vel


        msg.buttons = [self.bias, x, -y]
        self.cmd_vel_pub.publish(msg)

        self.pre_x_error = x_error
        self.pre_y_error = y_error


    def odom_callback(self, msg):


        self.x0 = msg.pose.pose.position.x
        self.y0 = msg.pose.pose.position.y
        _, _, self.yaw0 = euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.odom_received = True
        

        if self.initialize is True:
            self.goal_des=[self.x0, self.y0, self.yaw0]
            self.initialize=False



if __name__ == '__main__':
    try:
        GoToGoal(nodename="go_to_goal")
    except rospy.ROSInterruptException:
        rospy.loginfo("Go to goal finished.")



