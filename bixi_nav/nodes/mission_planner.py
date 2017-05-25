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
    x0, y0, yaw0= 0.210, -0.265, 0

    #motion tolerance
    translation_tolerance=0.3
    angle_tolerance=5*math.pi/180
    
    #stores ideal positions of boxes
    pushing_pos=[[0.600, -0.480], [0.600, -0.800], [0.600, -1.160], [0.600, -1.520], [0.600, -1.880], [0.600, -2.140], [0.600, -2.600], [0.600, -2.960]]
    stacking_pos=[[3.750, -2.230], [3.050, -2.230], [2.550, -2.230], [3.750, -1.530], [3.050, -1.530], [3.750, -0.930], [3.050, -0.930], [2.550, -0.930]]

    desired_heading=0 #desired hole normal is opposite
    stacking_heading=desired_heading-math.pi/2

    #stores detected boxes
    box_centers=[]
    clustered_box_centers=[]
    box_headings=[]
    clustered_box_headings=[]
    
    #limit switch
    limit_switch=False

    #offsets from robot's body frame
    push_off=[0.38, -0.25]
    stack_off=[0.453, 0.078]

    previous_n_cluster=0
    n_cluster_counter=0

    #number of stacks to be lifted, e.g. 3->6
    n_boxes=3

    #ir values, 1 if sth detected, 0 if not
    ir_push=[0, 0]
    ir_stack=[0, 0]

    #counter
    first_stack=True

    def __init__(self, nodename):        
        rospy.init_node('mission_planner')
        #initialise ideal positions 

        rospy.Subscriber("/odometry", Odometry, self.odom_callback, queue_size = 50)
        #rospy.Subscriber("/actuation/limit_switch", Bool, self.limit_switch_callback, queue_size = 10)
        rospy.Subscriber("/edge", PoseStamped, self.edge_callback, queue_size=20)
        rospy.Subscriber("/actuation/ir_msg", Twist , self.ir_callback, queue_size=10)


        self.target_goal_pub=rospy.Publisher("/target_goal", PoseStamped, queue_size=10)
        #self.stack_pub=rospy.Publisher("/stack", Bool, queue_size=5)
        self.cmd_vel_pub=rospy.Publisher("/vel_cmd", Joy, queue_size=10)
        #self.box_pose_pub=rospy.Publisher("/box_center", PoseStamped, queue_size=20)


        push_index=0
        stack_index=0

        while not rospy.is_shutdown():
            #print(self.clustered_box_centers)
            #if still have boxes to be pushed, push according to list

            #print(self.ir_push)

            if push_index<len(self.pushing_pos)-1:    
                self.push_box(self.pushing_pos[push_index],  self.stacking_pos[push_index], True)
                push_index+=1
            if push_index==len(self.pushing_pos)-1:
                self.push_box(self.pushing_pos[push_index],  self.stacking_pos[push_index], False)
            # else:
            #     #if no more boxes to be pushed, perform stacking 
            #     if stack_index<self.n_boxes:
            #         if stack_index==0:
            #             self.stack_box(self.stacking_pos[stack_index], second_box=False, with_return=False)
            #         else:
            #             if stack_index%2==0:
            #                 self.stack_box(self.stacking_pos[stack_index], second_box=True, with_return=False)
            #             else:
            #                 self.stack_box(self.stacking_pos[stack_index], second_box=True, with_return=True)
            #         stack_index+=1
            #     else:
            #         #return to origin
            #         self.go_to_goal([0, 0])
            #         break
            
            rospy.sleep(0.1)

    def push_box(self, est_pos, dest_pos, with_return=True):

        goals=[]

        #1. align to an offset from expected
        d=0.5
        goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        if self.first_stack == False:
            self.go_to_goal(goals[0])
        
        #2. match laser detected boxes, align with real position at offset
        k=self.match(est_pos)
        d=0.4
        print(k)
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off))
        if self.first_stack == False:
            self.go_to_goal(goals[1])

        #3. finely adjust and engage box by moving towards it
        if self.first_stack == False:
            self.push_adjust(k, est_pos)
        else:
            self.first_stack=False
        #4. push forward
        d=-1.15      
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(self.desired_heading), k[0][1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        self.go_to_goal(goals[2])        

        #5. diagonal to an offset from destination
        d=0.5
        goals.append(self.offset_to_center([dest_pos[0]-d*math.cos(self.desired_heading), dest_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        self.go_to_goal(goals[3])  

        #6. forward until desired position
        goals.append(self.offset_to_center([dest_pos[0], dest_pos[1]], self.desired_heading, self.push_off))
        self.go_to_goal(goals[4])   

        #7. backward to an offset
        d=0.5
        goals.append(self.offset_to_center([dest_pos[0]-d*math.cos(self.desired_heading), dest_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        self.go_to_goal(goals[5])       

        #return if needed
        if with_return is True:
            #8. diagonal back to position 4
            self.go_to_goal(goals[2])
            #9. back to position 1
            self.go_to_goal(goals[0])


    def push_adjust(self, last_k, est_pos):
        print(self.ir_push)
        k=last_k
        #0 means free, element 0 is left, element 1 is right
        while self.ir_push[0]!=0 or self.ir_push[1]!=0 and not rospy.is_shutdown():
            print("adjust")
            print(self.ir_push)
            #get the vy from ir, while yawing direction from k
            if self.ir_push[0]!=0 and self.ir_push[1]==0:
                print("adjusting to left")
                #left not free, move left in k[1] direction
                d=0.03
                goal=[self.x0-d*math.sin(k[1]), self.y0+d*math.cos(k[1]), k[1]]
                self.go_to_goal(goal)
            elif self.ir_push[0]==0 and self.ir_push[1]!=0:
                #right not free, move right in k[1] direction
                print("adjusting to right")
                d=0.03
                goal=[self.x0+d*math.sin(k[1]), self.y0-d*math.cos(k[1]), k[1]]
                self.go_to_goal(goal)
            else:
                #realign with lidar
                goal=self.offset_to_center([self.x0, self.y0], k[1], self.push_off)
                self.go_to_goal(goal)

            #update k
            new_k=self.match(est_pos)
            if new_k[0][0]!=est_pos[0]:
                k=new_k

        print("aligned")
        #after aligned, engage box
        d=0.4
        goal=[self.x0+d*math.cos(k[1]), self.y0+d*math.sin(k[1]), k[1]]
        self.go_to_goal(goal)


    def stack_box(self, est_pos, second_box=True, with_return=True):

        goals=[]

        #1. align to an offset from expected
        d=0.5
        goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.stacking_heading, self.stack_off))
        self.go_to_goal(goals[0])
        
        #2. match laser detected boxes, align with real position at offset
        k=self.match(est_pos)
        d=0.4
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
        self.go_to_goal(goals[1])

        #3. engage box by moving towards it
        goals.append(self.offset_to_center([k[0][0], k[0][1]], k[1]-math.pi/2, self.stack_off))
        self.go_to_goal(goals[2])

        #4. push forward until box is lifted
        d=-0.2      
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
        self.go_to_goal(goals[3])

        if second_box == True:
            #grab the second box
            #5. match laser detected boxes, align with real position at offset
            d=-0.3
            k=self.match([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)])
            d=0.3
            goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
            self.go_to_goal(goals[4])

            #6. engage box by moving towards it
            goals.append(self.offset_to_center([k[0][0], k[0][1]], k[1]-math.pi/2, self.stack_off))
            self.go_to_goal(goals[5])

            #7. push forward until box is lifted
            d=-0.2      
            goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
            self.go_to_goal(goals[6])        
         
        #return if needed
        if with_return == True:
            #8. back to position 1
            self.go_to_goal(goals[0])



    def offset_to_center(self, position, heading, offset):
        #calculate where the center of robot must be given an offset position
        center_pose=[position[0]-offset[0]*math.cos(heading)+offset[1]*math.sin(heading), position[1]-offset[0]*math.sin(heading)-offset[1]*math.cos(heading), heading]
        return center_pose

    def limit_switch_callback(self, msg):
        self.limit_switch = msg.data

    def match(self, box_pos):
        #find nearest real box with respect to expected position
        # and check if within tolerance
        
        tolerance=0.3
        min_length=100
        box_found=False
        index=None

        #while len(self.clustered_box_centers)!=len(self.clustered_box_headings) or len(self.clustered_box_centers)==0 or len(self.clustered_box_headings)==0:
        #    rospy.sleep(1)

        for i in range(len(self.clustered_box_headings)):
            d=math.sqrt((self.clustered_box_centers[i][0]-box_pos[0])**2+(self.clustered_box_centers[i][1]-box_pos[1])**2)

            if d<min_length:
                if d<tolerance:
                    index=i
                min_length=d
        
        print(self.clustered_box_centers)

        if index==None or index>=len(self.clustered_box_centers) or index>=len(self.clustered_box_headings):
            #if box is not seen
            return box_pos, self.desired_heading
        else:            
            return self.clustered_box_centers[index], self.clustered_box_headings[index]

    def ir_callback(self, msg):


        if msg.linear.x>230:
            self.ir_push[0]=1
        else:
            self.ir_push[0]=0

        if msg.linear.y>230:
            self.ir_push[1]=1
        else:
            self.ir_push[1]=0


    def edge_callback(self, msg):
        n_edge=40
        #for a detected edge, add it into the list. If list is full, replace the first element.
        if len(self.box_centers)==n_edge:
            #remove the first element
            del self.box_centers[0]
            del self.box_headings[0]

        _, _, yaw_angle = euler_from_quaternion((msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))
        #4print(self.get_heading(yaw_angle))
        self.box_centers.append([msg.pose.position.x, msg.pose.position.y])
        self.box_headings.append(self.get_heading(yaw_angle))

        #perform clustering to edges
        X=np.asarray(self.box_centers)
        db = DBSCAN(eps=0.5, min_samples=5).fit(X)
        
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_==0:
            return

        #if no of clusters less than previous, don't update and iterate count
        if n_clusters_<self.previous_n_cluster:
            self.n_cluster_counter+=1
            if self.n_cluster_counter>15:
                self.previous_n_cluster=n_clusters_
                self.n_cluster_counter=0
                #print(n_clusters_)
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
        print("new goal:")
        print(goal)
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
                rospy.sleep(2)
                break
        print(self.x0, self.y0)

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
