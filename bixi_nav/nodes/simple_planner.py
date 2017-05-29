#!/usr/bin/env python  
import roslib
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped
from sensor_msgs.msg import Joy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, KMeans
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
import numpy as np
import math


class MissionPlanner(object):
    x0, y0, yaw0= 0, 0, 0

    #motion tolerance
    translation_tolerance=0.4
    angle_tolerance=3*math.pi/180
    
    #stores ideal positions of boxes
    pushing_pos=[[0.600, 0.630], [0.600, 1.080], [0.600, 1.530], [0.600, 1.980], [0.600, 2.430]]#, [0.600, -2.260], [0.600, -2.590], [0.600, -2.950]]
    stacking_pos=[[2.600, 0.630], [2.600, 1.080], [2.600, 1.530], [2.600, 1.980], [2.600, 2.430]]#, [3.850, -0.930], [3.050, -0.930], [2.550, -0.930]]

    extra_pushing_pos=[[0.800, 2.650], [0.600, 2.900], [0.800, 0.100]]
    extra_stacking_pos=[[3.600, 2.430], [3.600, 1.980], [3.600, 1.080]]

    desired_heading=0 #desired hole normal is opposite

    #stores detected boxes
    box_centers=[]
    clustered_box_centers=[]
    box_headings=[]
    clustered_box_headings=[]
    
    #limit switch-> true if inside, job_status->true if stacking done
    limit_sense=False
    job_status=False

    #offsets from robot's body frame
    push_off=[0.6, 0]
    stack_off=[0.430, 0.078]

    previous_n_cluster=0
    n_cluster_counter=0

    #number of stacks to be lifted, e.g. 3->6
    n_boxes=3

    #ir values, 1 if sth detected, 0 if not
    ir_stack=[0, 0]


    def __init__(self, nodename):        
        rospy.init_node('mission_planner')
        #initialise ideal positions 

        rospy.Subscriber("/odometry", Odometry, self.odom_callback, queue_size = 50)
        rospy.Subscriber("/edge", PoseStamped, self.edge_callback, queue_size=20)
        rospy.Subscriber("/actuation/ir_msg", Twist , self.ir_callback, queue_size=10)
        rospy.Subscriber("/actuation/limit_sense", Bool, self.limit_switch_callback, queue_size = 5)
        rospy.Subscriber("/actuation/job_status", Bool, self.job_status_callback, queue_size = 10)

        self.target_goal_pub=rospy.Publisher("/target_goal", PoseStamped, queue_size=10)
        self.cmd_vel_pub=rospy.Publisher("/vel_cmd", Joy, queue_size=10)
        self.disengage_pub=rospy.Publisher("/disengage", Bool, queue_size=1)

        push_index=0
        stack_index=0

        while not rospy.is_shutdown():
            #print(self.clustered_box_centers)
            #if still have boxes to be pushed, push according to list

            #print(self.ir_push)

            if push_index<len(self.pushing_pos):
                print("pushing box number {}".format(push_index+1))
                if push_index<len(self.pushing_pos)-1:    
                    self.push_box(self.pushing_pos[push_index],  self.stacking_pos[push_index], True)
                    push_index+=1
                elif push_index==len(self.pushing_pos)-1:
                    self.push_box(self.pushing_pos[push_index],  self.stacking_pos[push_index], False)
                    push_index+=1
            else:
                #if no more boxes to be pushed, perform stacking
                print("stacking box number {}".format(stack_index+1)) 
                if stack_index<self.n_boxes:
                    if stack_index==0:
                        #the first stack needs only to be stacked once
                        self.stack_box(self.stacking_pos[len(self.stacking_pos)-1-stack_index], False, False)
                    else:
                        #subsequent needs to be stacked twice
                        self.stack_box(self.stacking_pos[len(self.stacking_pos)-1-stack_index], True, True)
                    stack_index+=1
                else:
                    #disengage, wait for stepper to release
                    rospy.sleep(10)
                    #slowly drive sideway
                    #yaw correctly
                    self.go_to_goal([self.x0, self.y0, self.desired_heading-math.pi/2])
                    for i in range(10):
                        rospy.sleep(1)
                        #0.05 m at a time
                        d=0.05
                        self.go_to_goal([self.x0-d, self.y0, self.desired_heading-math.pi/2])

                    d=0.5
                    self.go_to_goal([self.x0-d, self.y0, self.desired_heading-math.pi/2])    
                    #return to origin
                    self.go_to_goal([0.3, 1.5, self.desired_heading])
                    break
            
            rospy.sleep(0.1)


        push_index=0
        while not rospy.is_shutdown():
            #push extra boxes if have time...
            if push_index<len(self.extra_pushing_pos):
                print("pushing extra boxes number {}".format(push_index+1))
                    self.extra_push_box(self.extra_pushing_pos[push_index],  self.extra_stacking_pos[push_index], True)
                    push_index+=1
            
            rospy.sleep(0.1)

    def push_box(self, est_pos, dest_pos, with_return):
        # if self.first_stack==True:
        #     self.first_stack=False
        #     return

        goals=[]

        #1. align to an offset from expected
        d=0.5
        goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        self.go_to_goal(goals[0]) 

        #2. match laser detected boxes, align with real position at offset
        k=self.match(est_pos)        
        d=0.5
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[1]) 

        k=self.match(est_pos)
        d=0.2
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[2])

        #3. forward until desired position
        goals.append(self.offset_to_center([dest_pos[0], dest_pos[1]], self.desired_heading, self.push_off))
        self.go_to_goal(goals[3])   

        #return if needed
        if with_return == True:
            #4. back to position 1
            self.go_to_goal([goals[0][0], goals[0][1]-0.2, self.desired_heading])



    def stack_box(self, est_pos, second_box, with_return):
        goals=[]

        #1. align to an offset from expected
        d=0.4
        goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading-math.pi/2, self.stack_off))
        self.go_to_goal(goals[0])
        
        #2. match laser detected boxes, align with real position at offset
        k=self.match(est_pos)
        d=0.3
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
        self.go_to_goal(goals[1])
        
        while self.limit_sense==False:
            self.stack_adjust(k, est_pos)
            
        self.limit_sense=False

        #4. stop until box is lifted
        while self.job_status==False:
            self.stop()
            rospy.sleep(1)
            
        self.job_status=False


        if second_box == True:
            #grab the second box
            #1. align to an offset from expected
            est_pos=[est_pos[0]+0.2, est_pos[1]]
            d=0.4
            goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading-math.pi/2, self.stack_off))
            self.go_to_goal(goals[2])
            
            #2. match laser detected boxes, align with real position at offset
            k=self.match(est_pos)
            d=0.3
            goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
            self.go_to_goal(goals[3])
            
            while self.limit_sense==False:
                self.stack_adjust(k, est_pos)
                
            self.limit_sense=False

            #4. push forward until box is lifted
            while self.job_status==False:
                self.stop()
                rospy.sleep(1)
                
            self.job_status=False   
         
        #return if needed
        if with_return == True:
            #8. back to position 1
            self.go_to_goal(goals[0])


    def stack_adjust(self, last_k, est_pos):

        self.translation_tolerance=0.2

        k=last_k
        #0 means free, element 0 is left, element 1 is right
        while self.ir_stack[0]!=0 or self.ir_stack[1]!=0 and not rospy.is_shutdown():
            
            print("adjust {}".format(self.ir_stack))

            #get the vy from ir, while yawing direction from k
            if self.ir_stack[0]!=0 and self.ir_stack[1]==0:
                print("adjusting to left")
                #left not free, move left in k[1] direction
                d=0.03
                goal=[self.x0+d*math.cos(k[1]-math.pi/2), self.y0-d*math.sin(k[1]-math.pi/2), k[1]-math.pi/2]
                self.go_to_goal(goal)
            elif self.ir_stack[0]==0 and self.ir_stack[1]!=0:
                #right not free, move right in k[1] direction
                print("adjusting to right")
                d=0.03
                goal=[self.x0-d*math.sin(k[1]-math.pi/2), self.y0+d*math.sin(k[1]-math.pi/2), k[1]-math.pi/2]
                self.go_to_goal(goal)
            else:
                #realign with lidar
                goal=self.offset_to_center([self.x0, self.y0], k[1]-math.pi/2, self.push_off)
                self.go_to_goal(goal)

            #update k
            new_k=self.match(est_pos)
            if new_k[0][0]!=est_pos[0]:
                print("newly matched box")
                k=new_k

        print("aligned")
        #after aligned, engage box
        d=0.05
        goal=[self.x0+d*math.cos(k[1]), self.y0+d*math.sin(k[1]), k[1]-math.pi/2]
        self.go_to_goal(goal)

        self.translation_tolerance=0.5


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
        
        print("matched box position {}".format(self.clustered_box_centers))

        if index==None or index>=len(self.clustered_box_centers) or index>=len(self.clustered_box_headings):
            #if box is not seen
            return box_pos, self.desired_heading
        else:            
            return self.clustered_box_centers[index], self.clustered_box_headings[index]


    def extra_push_box(self, est_pos, dest_pos, with_return):
        # if self.first_stack==True:
        #     self.first_stack=False
        #     return

        goals=[]

        #1. align to an offset from expected
        d=0.5
        goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        self.go_to_goal(goals[0]) 

        #2. match laser detected boxes, align with real position at offset
        k=self.match(est_pos)        
        d=0.5
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[1]) 

        k=self.match(est_pos)
        d=0.2
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[2])

        #push forward abit
        d=dest_pos[0]-k[0][0]   
        goals.append(self.offset_to_center([dest_pos[0], k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[3])        

        #move backward abit
        d=0.3
        goals.append([self.x0-d, self.y0, self.desired_heading])
        self.go_to_goal(goals[4]) 

        #move sideway, direction depending dest[1]-est[1]
        if dest_pos[1]-est_pos[1]>0:
            #destination in left of robot, robot go to right
            d=0.5
            goals.append(self.offset_to_center([dest_pos[0], k[0][1]-d], self.desired_heading, self.push_off))
            self.go_to_goal(goals[5]) 
            #go to pushing position
            goals.append(self.offset_to_center([dest_pos[0], k[0][1]-d], self.desired_heading+math.pi/2, self.push_off))
            self.go_to_goal(goals[6]) 
            #push to desired position
            goals.append(self.offset_to_center([dest_pos[0], dest_pos[1]], self.desired_heading+math.pi/2, self.push_off))
            self.go_to_goal(goals[7]) 

        else:
            #destination in left of robot, robot go to right
            d=0.5
            goals.append(self.offset_to_center([dest_pos[0], k[0][1]+d], self.desired_heading, self.push_off))
            self.go_to_goal(goals[5]) 
            #go to pushing position
            goals.append(self.offset_to_center([dest_pos[0], k[0][1]+d], self.desired_heading-math.pi/2, self.push_off))
            self.go_to_goal(goals[6]) 
            goals.append(self.offset_to_center([dest_pos[0], dest_pos[1]], self.desired_heading-math.pi/2, self.push_off))
            self.go_to_goal(goals[7])

        #return maneuver
        self.go_to_goal(goals[6])
        self.go_to_goal(goals[0])



    def ir_callback(self, msg):

        if msg.angular.x>200:
            self.ir_stack[0]=1
        else:
            self.ir_stack[0]=0

        if msg.angular.y>200:
            self.ir_stack[1]=1
        else:
            self.ir_stack[1]=0

    def limit_switch_callback(self, msg):
        self.limit_sense = msg.data

    def job_status_callback(self, msg):
        self.job_status = msg.data

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
        print("new goal: {}".format(goal))
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

    def stop(self):
        msg=PoseStamped()
        msg.pose.position.z = 1
        self.target_goal_pub.publish(msg)

    def correct_range(self, theta):
        #warp angle into range of -pi, pi
        return math.atan2(math.sin(theta), math.cos(theta)) 

    def offset_to_center(self, position, heading, offset):
        #calculate where the center of robot must be given an offset position
        center_pose=[position[0]-offset[0]*math.cos(heading)+offset[1]*math.sin(heading), position[1]-offset[0]*math.sin(heading)-offset[1]*math.cos(heading), heading]
        return center_pose

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
