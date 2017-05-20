#!/usr/bin/env python  
import roslib
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped
from sensor_msgs.msg import Joy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, KMeans
from visualization_msgs.msg import Marker

import numpy as np
import math


class MissionPlanner(object):
    x0, y0, yaw0= 0, 0, 0

    #motion tolerance
    translation_tolerance=0.2
    angle_tolerance=5*math.pi/180
    
    #stores ideal positions of boxes
    pushing_pos=[[1.5, 1.5], [1.5, 0.8]]
    stacking_pos=[[2.5, 0.8], [2.5, -0.2]]

    desired_heading=0 #desired hole normal is opposite

    #stores detected boxes
    box_centers=[]
    clustered_box_centers=[]
    box_headings=[]
    clustered_box_headings=[]
    #limit switch
    limit_switch=False

    #offset for pushing
    push_off=[0.38, -0.15]

    #offset for stacking
    stack_off=[0.453, 0.078]

    previous_n_cluster=0
    n_cluster_counter=0


    def __init__(self, nodename):        
        rospy.init_node('mission_planner')
        #initialise ideal positions 

        rospy.Subscriber("/odometry", Odometry, self.odom_callback, queue_size = 50)
        #rospy.Subscriber("/limit_switch", Bool, self.limit_switch_callback, queue_size = 10)
        rospy.Subscriber("/edge", PoseStamped, self.edgeCallback, queue_size=20)

        self.target_goal_pub=rospy.Publisher("/target_goal", PoseStamped, queue_size=10)
        #self.stack_pub=rospy.Publisher("/stack", Bool, queue_size=5)
        self.cmd_vel_pub=rospy.Publisher("/vel_cmd", Joy, queue_size=10)
        #self.box_pose_pub=rospy.Publisher("/box_center", PoseStamped, queue_size=20)

        push_index=0
        stack_index=0

        while not rospy.is_shutdown():
            #if still have boxes to be pushed, push according to list
            if push_index<len(self.pushing_pos):    
                self.push_box(self.pushing_pos[push_index],  self.stacking_pos[push_index], True)
                push_index+=1

            # else:
            #     #if no more boxes to be pushed, perform stacking 
            #     if stack_index<2:
            #        self.stack_box(self.stacking_pos[stack_index])
            #        stack_index+=1
            #     else:
            #         #hopefully we've got 10 boxes
            #         break
            
            rospy.sleep(0.1)

    def push_box(self, est_pos, dest_pos, with_return=True):
        #go to some distance from estimated position of box
        d=0.5
        goal_1=self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off)
        print("goal 1")
        print(goal_1)
        self.go_to_goal(goal_1)
        

        d=0.4
        #match laser detected boxes
        k=self.match(est_pos)
        print(k)
        #align to real position        
        goal_2=self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off)
        print("goal 2")
        print(goal_2)
        self.go_to_goal(goal_2)

        goal_3=self.offset_to_center([k[0][0], k[0][1]], k[1], self.push_off)
        print("goal 3")
        print(goal_3)
        self.go_to_goal(goal_3)

        #push to a distance forward in the direction of box
        #k=self.match(est_pos)
        d=-0.8        
        goal_4=self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off)
        print("goal 4")
        print(goal_4)
        self.go_to_goal(goal_4)        

        d=0.5
        goal_5=self.offset_to_center([dest_pos[0]-d*math.cos(self.desired_heading), dest_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off)
        print("goal 5")
        print(goal_5)
        self.go_to_goal(goal_5)  

        #push to final destination position
        goal_6=self.offset_to_center([dest_pos[0], dest_pos[1]], self.desired_heading, self.push_off)
        print("goal 6")
        print(goal_6)
        self.go_to_goal(goal_6)   

        d=0.5
        goal_7=self.offset_to_center([dest_pos[0]-d*math.cos(self.desired_heading), dest_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off)
        print("goal 7")
        print(goal_7)
        self.go_to_goal(goal_7)       

        #return if needed
        if with_return is True:
            self.go_to_goal(goal_1)


    def offset_to_center(self, position, heading, offset):
        #calculate where the center of robot must be given an offset position
        center_pose=[position[0]-offset[0]*math.cos(heading)+offset[1]*math.sin(heading), position[1]-offset[0]*math.sin(heading)-offset[1]*math.cos(heading), heading]
        return center_pose

    def stack_box(self, est_pos):
        #go to some distance from estimated position of box
        d=0.4
        goal_1=[est_pos[0]+d*math.cos(self.desired_heading-math.pi/2), 
                est_pos[1]+d*math.sin(self.desired_heading-math.pi/2), 
                self.desired_heading]

        self.go_to_goal(goal_1)

        #match laser detected boxes
        k=self.match(est_pos)

        #align to real position
        d=0.2
        goal_2=[k[0][0]+d*math.cos(self.desired_heading-math.pi/2), 
                k[0][1]+d*math.sin(self.desired_heading-math.pi/2), 
                k[1]]
        self.go_to_goal(goal_2)



        #move until limit switches touches
        k=self.match(est_pos)

        #align to real position
        d=0.3
        goal_3=[k[0][0]+d*math.cos(self.desired_heading-math.pi/2), 
                k[0][1]+d*math.sin(self.desired_heading-math.pi/2), 
                k[1]]
        self.go_to_goal(goal_3)
        
        d=-0.1
        goal_4=[k[0][0]+d*math.cos(self.desired_heading-math.pi/2), 
                k[0][1]+d*math.sin(self.desired_heading-math.pi/2), 
                k[1]]
        self.go_to_goal(goal_4)

        rospy.sleep(10)

        #take another one on the side
        # real_pos=self.match(est_pos)

        # goal_4=[real_pos[0],
        #         real_pos[1],
        #         self.desired_heading]
        # self.go_to_goal(goal_4)
        # self.takein_box(goal_4)

    def takein_box(self):
        #move until limit switches touches
        while not True and not rospy.is_shutdown():
            msg=Twist()
            vel=150           
            msg.buttons=[1024, 1024+vel, 1024]
            self.cmd_vel_pub.publish(msg) 
            if self.limit_switch:
                break

        #initiate stacking, publish True to arduino
        msg=Bool()
        msg.data=True
        self.stack_pub.publish(True)

    def limit_switch_callback(self, msg):
        self.limit_switch = msg.data

    def match(self, box_pos):
        #find nearest real box with respect to expected position
        # and check if within tolerance
        
        tolerance=0.3
        min_length=100
        box_found=False
        index=0

        #while box_found is False:
        #    min_length=100
        while len(self.clustered_box_centers)!=len(self.clustered_box_headings):
            rospy.sleep(1)


        for i in range(len(self.clustered_box_headings)):
            d=math.sqrt((self.clustered_box_centers[i][0]-box_pos[0])**2+(self.clustered_box_centers[i][1]-box_pos[1])**2)

            if d<min_length:
                index=i
                min_length=d

        print(index)
        print(len(self.clustered_box_centers), len(self.clustered_box_headings))
        #    if index is None:
                #rotate
                #self.see_around()

            #     continue
            # else:
            #     if len(self.clustered_box_centers)<=index or len(self.clustered_box_headings)<=index:
            #         #rotate and match again
            #         #self.see_around()
            #         continue
            #     else:
            #         box_found=True
            #         break

        #if min_length<tolerance:            
        return self.clustered_box_centers[index], self.clustered_box_headings[index]
        #else:
        #    return 

    def see_around(self):
        goal=[self.x0, 
              self.y0, 
              self.yaw0+math.pi/6]
        self.go_to_goal(goal)

        goal=[self.x0, 
              self.y0, 
              self.yaw0-math.pi/6]
        self.go_to_goal(goal)

        goal=[self.x0, 
              self.y0, 
              self.yaw0]
        self.go_to_goal(goal)

    def edgeCallback(self, msg):
        n_edge=80
        #for a detected edge, add it into the list. If list is full, replace the first element.
        if len(self.box_centers)==n_edge:
            #remove the first element
            del self.box_centers[0]
            del self.box_headings[0]

        _, _, yaw_angle = euler_from_quaternion((msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))
        #4print(self.get_heading(yaw_angle))
        self.box_centers.append([msg.pose.position.x, msg.pose.position.y])
        self.box_headings.append(self.get_heading(yaw_angle))

        #print([msg.pose.position.x, msg.pose.position.y])

        #perform clustering to edges
        X=np.asarray(self.box_centers)
        db = DBSCAN(eps=0.5, min_samples=10).fit(X)
        
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_==0:
            return

        #if no of clusters less than previous, don't update and iterate count
        if n_clusters_<self.previous_n_cluster:
            self.n_cluster_counter+=1
            if self.n_cluster_counter>20:
                self.previous_n_cluster=n_clusters_
                self.n_cluster_counter=0
                print(n_clusters_)
            return
        else:
            self.previous_n_cluster=n_clusters_

        clusters = [X[labels == i] for i in range(n_clusters_)]

        heading_cluster=[]

        self.clustered_box_centers=[]
        self.clustered_box_headings=[]

        for i in range(len(clusters)):
            position_kmeans=KMeans(n_clusters=1).fit(clusters[i])
            position_center=position_kmeans.cluster_centers_
            #print(position_center)
            self.clustered_box_centers.append(position_center[0])
            heading_list=[]
        
            for j in range(len(self.box_headings)):
                if labels[j]==i:
                    heading_list.append(self.box_headings[j])
            mean_heading=np.mean(heading_list)
            heading_cluster.append(mean_heading)
        #print(heading_cluster)
        #print(self.clustered_box_headings)
        #print(self.clustered_box_centers)

        self.clustered_box_headings=heading_cluster
        

    def get_heading(self, direction):
        heading=self.correct_range(direction-self.desired_heading)

        if heading<math.pi/4 and heading>-math.pi/4:
            result=direction
        elif heading<3*math.pi/4 and heading>math.pi/4:
            result=direction-math.pi/2
        elif heading>-3*math.pi/4 and heading<-math.pi/4:
            result=direction+math.pi/2
        else:
            result=direction-math.pi

        return self.correct_range(result)


    def go_to_goal(self, goal):
        msg=PoseStamped()

        msg.header.frame_id="odom"
        msg.pose.position.x = goal[0]
        msg.pose.position.y = goal[1]
        q_angle = quaternion_from_euler(0, 0, goal[2])
        msg.pose.orientation = Quaternion(*q_angle)

        while not rospy.is_shutdown():
            #at least publish once, 
            self.target_goal_pub.publish(msg)
            #if goal achieved
            if math.sqrt((self.x0-goal[0])**2+(self.y0-goal[1])**2)<self.translation_tolerance**2 and abs(self.correct_range(self.yaw0-goal[2]))<self.angle_tolerance:
                break

    def correct_range(self, theta):
        #warp angle into range of -pi, pi
        return math.atan2(math.sin(theta), math.cos(theta)) 


    def printBox(self):

        local_position=self.clustered_box_centers
        local_headings=self.clustered_box_headings

        #print(len(self.clustered_box_headings), len(self.clustered_box_centers))
        for i in range(len(local_position)):
            #markerList store points wrt 2D world coordinate
            msg=PoseStamped()
            msg.header.frame_id="odom"
            msg.pose.position.x = local_position[i][0]
            msg.pose.position.y = local_position[i][1]
            q_angle = quaternion_from_euler(0, 0, local_headings[i])
            msg.pose.orientation = Quaternion(*q_angle)
            self.box_pose_pub.publish(msg)


    def odom_callback(self, msg):
        self.x0 = msg.pose.pose.position.x
        self.y0 = msg.pose.pose.position.y
        _, _, self.yaw0 = euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.odom_received = True
        

if __name__ == '__main__':
    try:
        MissionPlanner(nodename="mission_planner")
    except rospy.ROSInterruptException:
        rospy.loginfo("Mission planner finished")
