#!/usr/bin/env python
import roslib; roslib.load_manifest('project2')
import rospy
import rospkg
import sys
import re

from geometry_msgs.msg import Twist
from kobuki_msgs.msg import SensorState
from blobfinder2.msg import MultiBlobInfo3D

from transform2d import transform2d_from_ros_transform, distance_2d, Transform2D

import tf.transformations as transformations

import tf
import math
import numpy as np

import solution

# sets control frequency
CONTROL_PERIOD = rospy.Duration(1.0/100.0)

# minimum duration of safety stop (s)
STOP_DURATION = rospy.Duration(1.0)

# minimum distance of cones to make gate (m)
MIN_GATE_DIST = 0.55

# maximum distance of cones to make gate (m)
MAX_GATE_DIST = 0.90

# minimum number of pixels to identify cone (px)
MIN_CONE_AREA = 200

# offsets to object colors
OBJECT_OFFSETS = dict(
    green_cone = 0.04,
    red_tape = 0.0,
    yellow_cone = 0.04
)

######################################################################
# read a course from a text description and return a list of string
# triples like ('left', 'green', 'yellow')

def read_course(course_file):
    
    VALID_ACTIONS = ['left', 'right', 'tape']
    VALID_COLORS = ['yellow', 'green']

    input_stream = open(course_file, 'r')

    rows = []

    comment = re.compile('#.*$')

    for line in input_stream:

        line = line.strip()

        line = re.sub(comment, '', line)

        if not len(line):
            continue
        
        tokens = line.lower().split()
        
        if len(tokens) != 3:
            rospy.logerr('syntax error in ' + course_file)
            sys.exit(1)
            
        action, lcolor, rcolor = tokens
        
        if action not in VALID_ACTIONS:
            rospy.logerr('unexpected action type ' + action + ' in ' + course_file)
            sys.exit(1)
            
        if lcolor not in VALID_COLORS or rcolor not in VALID_COLORS:
            rospy.logerr('unexpected color pair {} {} in {}'.format(
                lcolor, rcolor, course_file))
            sys.exit(1)

        row = (action, lcolor, rcolor)
        rows.append(row)

    if not len(rows):
        rospy.logerr('empty course file!')
        sys.exit(1)

    return rows
            
###################################################################### 
# define a class to handle our simple controller

class Controller:

    # initialize our controller
    def __init__(self):

        # initialize our ROS node
        rospy.init_node('starter')

        # read the course file and store it into self.course
        self.setup_course()

        # set up publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # transformation to robot initial frame from odom frame
        self.world_from_odom = None
        
        # set up a TransformListener to get odometry information
        self.odom_listener = tf.TransformListener()
        
        # record whether we should stop for safety
        self.should_stop = 0
        self.time_of_stop = rospy.get_rostime() - STOP_DURATION

        self.latest_blobs = dict()
        self.tape_angular_z = 0.0

        # No gate yet
        self.cur_gate = None

        # No tape yet
        self.cur_tape = None

        # put myself at start of course
        self.course_index = 0

        # Start at 0 velocity
        self.prev_cmd_vel = Twist()

        # Start going straight for tape
        self.tape_angular_z = 0.0

        # set up our trivial 'state machine' controller
        rospy.Timer(CONTROL_PERIOD,
                    self.control_callback)
        
        # set up subscriber for sensor state for bumpers/cliffs
        rospy.Subscriber('/mobile_base/sensors/core',
                         SensorState, self.sensor_callback)

        # set up subscriber for color blobs
        rospy.Subscriber('blobfinder2/blobs3d',
                         MultiBlobInfo3D,
                         self.blobs3d_callback)


    # set up the course from the file
    def setup_course(self):

        # read in the course file and store it into self.course        
        args = rospy.myargv(argv=sys.argv)

        if len(args) != 2:
            rospy.logerr('usage: project3.py COURSENAME.txt')
            sys.exit(1)

        course_file = args[1]

        if '/' not in course_file:
            rospack = rospkg.RosPack()
            project3_path = rospack.get_path('project3')
            course_file = project3_path + '/data/' + course_file

        self.course = read_course(course_file)

        rospy.loginfo('read course: '+repr(self.course))

    # called when sensor msgs received - just copy sensor readings to
    # class member variables
    def sensor_callback(self, msg):

        if msg.bumper & SensorState.BUMPER_LEFT:
            rospy.loginfo('***LEFT BUMPER***')
        if msg.bumper & SensorState.BUMPER_CENTRE:
            rospy.loginfo('***MIDDLE BUMPER***')
        if msg.bumper & SensorState.BUMPER_RIGHT:
            rospy.loginfo('***RIGHT BUMPER***')
        if msg.cliff:
            rospy.loginfo('***CLIFF***')

        if msg.bumper or msg.cliff:
            self.should_stop = True
            self.time_of_stop = rospy.get_rostime()
        else:
            self.should_stop = False

    def blobs3d_callback(self, msg):

        tmp_latest_blobs = dict()

        T_world_from_robot = self.get_current_pose()

        if T_world_from_robot is None:
            rospy.logwarn('no xform yet in blobs3d_callback')
            return

        # get a 4x4 transformation matrix to map
        # from camera frame to robot frame
        try:
            
            p, q = self.odom_listener.lookupTransform(
                '/base_footprint',
                '/camera_rgb_optical_frame',
                rospy.Time(msg.header.stamp.secs,
                           msg.header.stamp.nsecs))

            transform_matrix = transformations.quaternion_matrix(q)

            transform_matrix[:3, 3] = p

        except tf.LookupException:

            rospy.loginfo('no tf in blobs3d_callback :(')

            return

        closest_tape = None

        # for each color object
        for cblobs in msg.color_blobs:

            # get object name
            color = cblobs.color.data

            if color not in OBJECT_OFFSETS:
                continue

            # list of positions to add to
            positions = []

            # for each blob in message
            for blob3d in cblobs.blobs:

                # discard blobs without 3D position info
                if not blob3d.have_pos:
                    continue

                # transform point from camera optical frame
                # to robot base frame
                p_camera = np.array([blob3d.position.x,
                                     blob3d.position.y,
                                     blob3d.position.z,
                                     1.0])

                # only need x and y coords, so first two elements here
                p_robot = np.dot(transform_matrix, p_camera)[:2]

                # get length of position vector
                l = np.linalg.norm(p_robot)

                # normalize
                d = p_robot / l

                # add a bit of length to the object to account for the
                # fact that we only see the front and not the back
                p_robot += d * OBJECT_OFFSETS[color]

                p_world = T_world_from_robot * p_robot

                # add to position list
                positions.append(p_world)

            # add positions to dictionary
            if len(positions):
                tmp_latest_blobs[color] = np.array(positions)

        self.latest_blobs = tmp_latest_blobs
                
    # get current pose from TransformListener
    def get_current_pose(self):

        try:

            ros_xform = self.odom_listener.lookupTransform(
                '/odom', '/base_footprint',
                rospy.Time(0))

        except tf.LookupException:

            return None

        odom_from_robot_cur = transform2d_from_ros_transform(ros_xform)

        if self.world_from_odom is None:
            self.world_from_odom = odom_from_robot_cur.inverse()
            return Transform2D()
        else:
            world_from_robot_cur = self.world_from_odom * odom_from_robot_cur
            return world_from_robot_cur


    # called when we want to find an object
    def find_objects(self, cur_pose, lcolor, rcolor):

        latest_blobs = self.latest_blobs
        self.latest_blobs = dict()

        lcone = lcolor + '_cone'
        rcone = rcolor + '_cone'

        #rospy.loginfo('%s %s %s', latest_blobs, lcone, rcone)

        # see if we can update the current gate
        if lcone in latest_blobs and rcone in latest_blobs:

            #rospy.loginfo('got cone pair')
            
            for lxy_world in latest_blobs[lcone]:
                
                lxy_robot = cur_pose.transform_inv(lxy_world)
                
                for rxy_world in latest_blobs[rcone]:

                    rxy_robot = cur_pose.transform_inv(rxy_world)

                    if lxy_robot[1] > rxy_robot[1]:
                        
                        diff_world = lxy_world - rxy_world
                        len = np.linalg.norm(diff_world)

                        if (len >= MIN_GATE_DIST and len <= MAX_GATE_DIST):

                            midpoint = 0.5 * (lxy_world + rxy_world)

                            theta = np.arctan2(-diff_world[0],
                                               diff_world[1])

                            self.cur_gate = Transform2D(midpoint[0],
                                                        midpoint[1],
                                                        theta)
                            
                            rospy.loginfo('****** updated %s %s gate! ******', lcolor, rcolor)

        # see if we can update the current tape
        if 'red_tape' in latest_blobs:

            best_dist = None

            for txy_world in latest_blobs['red_tape']:

                txy_robot = cur_pose.transform_inv(txy_world)

                # make sure in front of robot
                if txy_robot[0] > 0.0:

                    dist = np.linalg.norm(txy_robot)
                    
                    if best_dist is None or dist < best_dist:

                        best_dist = dist
                        self.cur_tape = txy_world

    # called periodically to do top-level coordination of behaviors
    def control_callback(self, timer_event=None):

        # initialize vel to 0, 0
        cmd_vel = Twist()

        time_since_stop = rospy.get_rostime() - self.time_of_stop

        cur_pose = self.get_current_pose()

        if cur_pose is None:
            
            rospy.loginfo('waiting for odometry...')

        elif self.should_stop or time_since_stop < STOP_DURATION:
            
            rospy.loginfo('stopped')

        else: # navigate the course

            if self.course_index >= len(self.course):
            
                rospy.loginfo('done!')
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0

                if (self.prev_cmd_vel.linear.x == 0.0 and
                    self.prev_cmd_vel.angular.z == 0.0):
                    rospy.signal_shutdown('done')

            else:
                
                # get the current line from the course
                cur_course = self.course[self.course_index]

                # update cur gate and cur_tape if possible
                self.find_objects(cur_pose, cur_course[1], cur_course[2])

                cmd_vel, cur_is_done = solution.navigate_course(cur_pose,
                                                                cur_course,
                                                                self.cur_gate,
                                                                self.cur_tape)

                if cur_is_done:

                    self.course_index += 1
                    
                    self.cur_gate = None
                    
                    self.cur_tape = None

            # now filter large changes in velocity before commanding
            # robot - note we don't filter when stopped
            cmd_vel.linear.x = solution.filter_linear_vel(self.prev_cmd_vel.linear.x,
                                                          cmd_vel.linear.x,
                                                          CONTROL_PERIOD.to_sec())
            
            cmd_vel.angular.z = solution.filter_angular_vel(self.prev_cmd_vel.angular.z,
                                                            cmd_vel.angular.z,
                                                            CONTROL_PERIOD.to_sec())

        self.cmd_vel_pub.publish(cmd_vel)
        
        self.prev_cmd_vel = cmd_vel

    # called by main function below (after init)
    def run(self):
        
        # timers and callbacks are already set up, so just spin
        rospy.spin()

        # if spin returns we were interrupted by Ctrl+C or shutdown
        rospy.loginfo('goodbye')


# main function
if __name__ == '__main__':
    try:
        ctrl = Controller()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
    
