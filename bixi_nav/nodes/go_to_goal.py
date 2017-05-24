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

    def __init__(self, nodename):
        heading_threshold=5*math.pi/180

        rospy.init_node('go_to_goal')

        rospy.Subscriber("/odometry", Odometry, self.odom_callback, queue_size = 50)
        rospy.sleep(1)
        rospy.Subscriber("/target_goal", PoseStamped, self.goal_callback , queue_size=10)

        self.cmd_vel_pub=rospy.Publisher("/vel_cmd", Joy, queue_size=10)

        r = rospy.Rate(10)

        while not rospy.is_shutdown():

            #if direction not similar, rotate
            ang_error=abs(math.atan2(math.sin(self.yaw0-self.goal_des[2]), math.cos(self.yaw0-self.goal_des[2])))
            if ang_error>heading_threshold:
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
        angular_vel=100 #max is 660

        ang_error=math.atan2(math.sin(angle-self.yaw0), math.cos(angle-self.yaw0))
        print(ang_error*180/math.pi)
        if ang_error>0:
            msg.buttons=[1024, 1024, 1024-angular_vel]
        else:
            msg.buttons=[1024, 1024, 1024+angular_vel]
        
        self.cmd_vel_pub.publish(msg)
        

    def translate(self, x_target, y_target):
        msg=Joy()
        vel=150 #must be small to avoid jerking, and secondly to avoid switching surface
        distance_threshold=0.05
        x_error=(x_target-self.x0)*math.cos(self.yaw0)+(y_target-self.y0)*math.sin(self.yaw0)
        y_error=-(x_target-self.x0)*math.sin(self.yaw0)+(y_target-self.y0)*math.cos(self.yaw0)
        print("translate")
        print(math.sqrt(x_error**2+y_error**2))
        print(x_target, y_target)

        if abs(x_error)>distance_threshold:
            if x_error>0:
                vx=1024+vel
            else:
                vx=1024-vel
        else:
            vx=1024

        if abs(y_error)>distance_threshold:
            if y_error>0:
                vy=1024-vel
            else:
                vy=1024+vel
        else:
            vy=1024

        msg.buttons=[vx, vy, 1024]
        
        self.cmd_vel_pub.publish(msg)


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

