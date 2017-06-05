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
    pushing_pos=[[0.800, 0.100], [0.800, 0.500], [0.800, 0.900], [0.800, 1.300], [0.800, 1.700], [0.800, 2.100], [0.800, 2.500], [0.800, 2.900]]
    stacking_pos=[[3.500, 0.600], [3.500, 0.600], [2.800, 0.900], [2.800, 1.300], [2.800, 1.700], [2.800, 2.100], [3.500, 2.500], [3.500, 2.500]]


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
    push_off=[0.5, 0.17]
    stack_off=[0.430, 0.078]

    previous_n_cluster=0
    n_cluster_counter=0

    #number of stacks to be lifted, e.g. 3->6
    n_boxes=3

    #ir values, 1 if sth detected, 0 if not
    ir_stack=[0, 0]

    #disengage
    disengage=False


    def __init__(self, nodename):        
        rospy.init_node('mission_planner')
        #initialise ideal positions 

        rospy.Subscriber("/odometry", Odometry, self.odom_callback, queue_size = 50)
        rospy.Subscriber("/edge", PoseStamped, self.edge_callback, queue_size=20)
        # rospy.Subscriber("/actuation/ir_msg", Twist , self.ir_callback, queue_size=10)
        # rospy.Subscriber("/actuation/limit_sense", Bool, self.limit_switch_callback, queue_size = 5)
        # rospy.Subscriber("/actuation/job_status", Bool, self.job_status_callback, queue_size = 10)

        self.target_goal_pub=rospy.Publisher("/target_goal", PoseStamped, queue_size=10)
        self.cmd_vel_pub=rospy.Publisher("/vel_cmd", Joy, queue_size=10)
        self.disengage_pub=rospy.Publisher("/actuation/disengage", Bool, queue_size=1)

        push_index=0


        while not rospy.is_shutdown():
            #print(self.clustered_box_centers)
            #if still have boxes to be pushed, push according to list

            #print(self.ir_push)

            if push_index<len(self.pushing_pos):
                print("pushing box number {}".format(push_index+1))

                if push_index==0:
                    print("hello")
                    self.go_to_goal([self.x0-0.1, self.y0, self.desired_heading])

                self.push_box(self.pushing_pos[push_index],  self.stacking_pos[push_index], True)
                push_index+=1

            else:
                break

        print("pushing planner done")

    def push_box(self, est_pos, dest_pos, with_return):


        goals=[]

        #1. align to an offset from expected
        d=0.3
        goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading, self.push_off))
        self.go_to_goal(goals[0]) 

        #2. match laser detected boxes, align with real position at offset
        k=self.match(est_pos)        
        d=0.3
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[1]) 

        k=self.match(est_pos)
        d=0.15
        goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], self.desired_heading, self.push_off))
        self.go_to_goal(goals[2])

        if k[0][1]<0.49:
            #push forward, yaw 30 degree, and push forward until is inside box 
            goals.append(self.offset_to_center([k[0][0]+2.5, k[0][1]], self.desired_heading, self.push_off))
            self.go_to_goal(goals[3])   


            self.go_to_goal([self.x0, self.y0-0.5, self.desired_heading])

            self.go_to_goal([self.x0+0.3, self.y0, self.desired_heading])

            self.go_to_goal([self.x0, self.y0+1, self.desired_heading])

            self.go_to_goal([self.x0, self.y0-0.3, self.desired_heading])

            # goal=self.offset_to_center([k[0][0]+1.8, k[0][1]], self.desired_heading+math.pi/6, self.push_off)
            # self.go_to_goal(goal)

            # goal=self.offset_to_center([k[0][0]+2.3, k[0][1]+0.5], self.desired_heading+math.pi/6, self.push_off)
            # self.go_to_goal(goal)

            # k=self.match([self.x0+0.6, self.y0])
            # d=0.1
            # self.go_to_goal(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off))

            # d=-0.7
            # self.go_to_goal(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off))



        elif k[0][1]>0.49 and k[0][1]<2.40:
            #3. forward until desired position
            goals.append(self.offset_to_center([dest_pos[0], dest_pos[1]], self.desired_heading, self.push_off))
            self.go_to_goal(goals[3])   

        elif k[0][1]>2.40:
            #push forward, yaw -30 degree, and push forward until is inside box 
            goals.append(self.offset_to_center([k[0][0]+2.5, k[0][1]], self.desired_heading, self.push_off))
            self.go_to_goal(goals[3])   

            #go left
            self.go_to_goal([self.x0, self.y0+0.6, self.desired_heading])

            self.go_to_goal([self.x0+0.3, self.y0, self.desired_heading])

            self.go_to_goal([self.x0, self.y0-0.8, self.desired_heading])

            self.go_to_goal([self.x0, self.y0+0.3, self.desired_heading])
            # goal=self.offset_to_center([k[0][0]+1.8, k[0][1]], self.desired_heading-math.pi/6, self.push_off)
            # self.go_to_goal(goal)1

            # goal=self.offset_to_center([k[0][0]+2.3, k[0][1]-0.5], self.desired_heading-math.pi/6, self.push_off)
            # self.go_to_goal(goal)

            # k=self.match([self.x0+0.6, self.y0])
            # d=0.1
            # self.go_to_goal(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off))

            # d=-0.7
            # self.go_to_goal(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1], self.push_off))

            # goal=self.offset_to_center([k[0][0]+3.0, k[0][1]-0.6], self.desired_heading-math.pi/6, self.push_off)
            # self.go_to_goal(goal)


        #return if needed
        if with_return == True:
            #4. back to position 1
            self.go_to_goal([self.x0-0.3, self.y0, self.desired_heading])
            self.go_to_goal([goals[3][0]-0.5, goals[3][1]-0.2, self.desired_heading])
            self.go_to_goal([goals[1][0], goals[1][1]-0.3, self.desired_heading])



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
        


        self.loop_adjust(k, est_pos)
        self.limit_sense=False

        while self.job_status==False and not rospy.is_shutdown():
            rospy.sleep(1)

        self.job_status=True

        if second_box == True:
            if self.disengage==True:
                #publish to tell it's the last box
                msg=Bool()
                msg.data=True
                self.disengage_pub(msg)   

            # #grab the second box
            # #1. align to an offset from expected
            # est_pos=[est_pos[0]+0.2, est_pos[1]]
            # d=0.2
            # goals.append(self.offset_to_center([est_pos[0]-d*math.cos(self.desired_heading), est_pos[1]-d*math.sin(self.desired_heading)], self.desired_heading-math.pi/2, self.stack_off))
            # self.go_to_goal(goals[2])
            
            # #2. match laser detected boxes, align with real position at offset
            # k=self.match(est_pos)
            # d=0.2
            # goals.append(self.offset_to_center([k[0][0]-d*math.cos(k[1]), k[0][1]-d*math.sin(k[1])], k[1]-math.pi/2, self.stack_off))
            # self.go_to_goal(goals[3])
            
            while self.limit_sense==False and not rospy.is_shutdown():
                self.stack_adjust(k, est_pos)
                
            self.limit_sense=False

            while self.job_status==False and not rospy.is_shutdown():
                rospy.sleep(1)

            self.job_status=True


         
        #return if needed
        if with_return == True:
            #8. back to position 1
            self.go_to_goal(goals[0])


    def stack_adjust(self, last_k, est_pos):

        self.translation_tolerance=0.15


        k=last_k

        alpha=k[1]-math.pi/2
        beta=k[1]-math.pi

        #0 means free, element 0 is left, element 1 is right
        while self.ir_stack[0]!=0 or self.ir_stack[1]!=0 and not rospy.is_shutdown():
            
            print("adjust {}".format(self.ir_stack))

            alpha=k[1]-math.pi/2
            beta=k[1]-math.pi

            #get the vy from ir, while yawing direction from k
            if self.ir_stack[0]!=0 and self.ir_stack[1]==0:
                print("adjusting to back")
                #left not free, move left in k[1] direction
                d=0.03
                goal=[self.x0, self.y0-d*math.sin(alpha), alpha]
                self.go_to_goal(goal)
            elif self.ir_stack[0]==0 and self.ir_stack[1]!=0:
                #right not free, move right in k[1] direction
                print("adjusting to front")
                d=0.03
                goal=[self.x0, self.y0+d*math.sin(alpha), alpha]
                self.go_to_goal(goal)

            elif self.ir_stack[0]==2 or self.ir_stack[1]==2:
                print("move further")
                d=0.01
                goal=[self.x0+d*math.cos(beta), self.y0+d*math.sin(beta), alpha]
                self.go_to_goal(goal)   
            else:
                #realign with lidar
                goal=[self.x0-0.05, self.y0, alpha]
                self.go_to_goal(goal)

            #update k
            new_k=self.match(est_pos)
            if new_k[0][0]!=est_pos[0]:
                print("newly matched box")
                k=new_k



        print("aligned")
        #after aligned, engage box
        d=0.10
        goal=[self.x0-d*math.cos(beta), self.y0-d*math.sin(beta), alpha]
        self.go_to_goal(goal)

        self.translation_tolerance=0.5


    def match(self, box_pos):
        #find nearest real box with respect to expected position
        # and check if within tolerance
        
        tolerance=0.2
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



    def ir_callback(self, msg):

        if msg.angular.x>250 and msg.angular.x<500:
            self.ir_stack[0]=1
        elif msg.angular.x>500:
            self.ir_stack[0]=2
        else:
            self.ir_stack[0]=0

        if msg.angular.y>250 and msg.angular.y<500:
            self.ir_stack[1]=1
        elif msg.angular.y>500:
            self.ir_stack[1]=2
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
