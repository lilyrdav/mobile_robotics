#!/usr/bin/env python

######################################################################
#
# project2.py
#
# Note you do not need to modify this file! Instead, modify
# solution.py in this directory.
#
# You can however read this if you want to see what the robot
# control code does outside of your own implementation tasks.
#
######################################################################

import roslib; roslib.load_manifest('project2')
import rospy
import rospkg
import sys

from geometry_msgs.msg import Twist
from kobuki_msgs.msg import SensorState
from blobfinder2.msg import MultiBlobInfo3D

from transform2d import transform2d_from_ros_transform, Transform2D

from tf import transformations

import tf
import math
import numpy as np

import solution

# control at 30 Hz
CONTROL_PERIOD = rospy.Duration(1.0/30.0)

OBJECT_OFFSETS = dict(
    blue_ball = 0.08,
    green_cone = 0.04
)

BALL_SEP_DIST = 0.4
KICK_DIST = 0.5

MIN_GOAL_WIDTH = 0.4
MAX_GOAL_WIDTH = 1.2

MIN_BALL_UPDATE_DIST = 0.5

POS_UPDATE_TOL = 0.05

MAX_LINEAR_X = 0.25
MAX_ANGULAR_Z = 1.5

######################################################################
# helper function to prevent large controls

def clip(name, cur_value, max_value):

    new_value = cur_value

    if cur_value > max_value:
        new_value = max_value
    elif cur_value < -max_value:
        new_value = -max_value

    if cur_value != new_value:
        rospy.loginfo('clipped %s from %f to %f',
                      name, cur_value, new_value)

    return new_value

######################################################################
# our controller class

class Controller:

    ######################################################################
    # constructor

    def __init__(self):
        rospy.init_node('starter')
        
        # set up publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # transformation to robot initial frame from odom frame
        self.world_from_odom = None

        # set up a TransformListener to get odometry information
        self.odom_listener = tf.TransformListener()

        # we will stop driving if picked up or about to drive off edge
        # of something
        self.cliff_alert = 0

        # transformation from goal frame to world frame
        self.world_from_goal = None

        # ball position in world coordinates
        self.ball_pos_world = None

        # "kick" position in world coordinates
        self.kick_pos_world = None

        # set up subscriber for sensor state for cliff sensors
        rospy.Subscriber('/mobile_base/sensors/core',
                         SensorState, self.sensor_callback)

        self.latest_blobs = dict()

        # subscribe to blob messages
        rospy.Subscriber('blobfinder2/blobs3d', MultiBlobInfo3D,
                         self.blobs3d_callback)

        # set current state
        self.state = 'init'

        # set up control timer at 100 Hz
        rospy.Timer(CONTROL_PERIOD, self.control_callback)

    ######################################################################
    # called whenever sensor messages are received
    
    def sensor_callback(self, msg):

        # set cliff alert
        self.cliff_alert = msg.cliff

    ######################################################################
    # stores robot-frame positions of objects in
    # a class member variable self.latest_blobs
    #
    # latest_blobs is a dictionary with keys that are object names
    # (e.g. 'green_cone') and values that are numpy n-by-3 arrays
    # that are lists of robot-frame (x, y) points
    
    def blobs3d_callback(self, msg):

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

        # reset latest_blobs to empty dictionary
        self.latest_blobs = dict()

        # for each color object
        for cblobs in msg.color_blobs:

            # get object name
            color = cblobs.color.data

            # ignore red_tape or yellow_cone
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

                # add to position list
                positions.append(p_robot)

            # add positions to dictionary
            if len(positions):
                self.latest_blobs[color] = np.array(positions)

    ######################################################################
    # returns information about the closest goal to the robot
    #
    # input: world_from_robot is the current robot pose as a Transform2D
    #        cone_positions is list of detected cones in robot frame
    #
    # output: None if no goal found, or tuple of goal information
    #         (world_from_goal, width, dist) 
    
    def find_closest_goal(self, world_from_robot, cone_positions):

        # variable we will return
        closest_info = None

        # for each cone pos
        for i, pi in enumerate(cone_positions):

            # for each other cone pos
            for pj in cone_positions[i+1:]:

                # order cone positions left and right
                if pi[1] >= pj[1]:
                    pl, pr = pi, pj
                else:
                    pl, pr = pj, pi

                # make sure left pos has greater y coord
                assert pl[1] >= pr[1]

                # get difference left-to-right
                diff = pl - pr

                # get the width of the potential goal
                width = np.linalg.norm(diff)

                # if it is in the correct range
                if width >= MIN_GOAL_WIDTH and width <= MAX_GOAL_WIDTH:

                    # get the midpoint
                    pmid = 0.5 * (pl + pr)

                    # get distance from robot to midpoint
                    dist = np.linalg.norm(pmid)

                    # see if we have a new closest goal
                    if closest_info is None or dist < closest_info[-1]:
                    
                        # get the forward vector of the goal in the robot frame
                        dx, dy = diff
                        fx, fy = dy, -dx

                        assert fx >= 0

                        # compute the angle of the forward vector relative
                        # to robot x axis
                        theta = np.arctan2(fy, fx)

                        # create a 2D rigid transform representing
                        # the goal pose in the robot frame
                        robot_from_goal = Transform2D(pmid[0], pmid[1], theta)

                        # now make a 2D rigid transform representing
                        # the goal pose in the world frame
                        world_from_goal = world_from_robot * robot_from_goal

                        # update the return value
                        closest_info = (world_from_goal, width, dist)

        # return info about closest goal we found
        return closest_info

    ######################################################################
    # find the position of the closest ball to the robot
    
    def find_closest_ball(self, world_from_robot, ball_positions):

        ball_dists = np.linalg.norm(ball_positions, axis=1)

        closest_idx = ball_dists.argmin()

        if ball_dists[closest_idx] < MIN_BALL_UPDATE_DIST:
            return None

        ball_robot = ball_positions[closest_idx]

        ball_pos_world = world_from_robot * ball_robot

        return ball_pos_world
                    
    ######################################################################
    # get current pose relative to start pose from TransformListener
    
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

    ######################################################################
    # update estimates of goal and ball position.
    #
    # this can modify several member variables:
    #
    #   self.world_from_goal      goal pose
    #
    #   self.ball_pos_world       position of ball in world frame
    #
    #   self.kick_pos_world       position in world frame where robot
    #                             needs to line up to kick the ball
    
    def update_world_model(self, cur_pose):

        # see if we need to update goal or ball pos
        update_goal = False
        update_ball = False            

        # check for new goal pose
        if 'green_cone' in self.latest_blobs:

            # get closest goal
            goal_info = self.find_closest_goal(cur_pose,
                                               self.latest_blobs['green_cone'])

            # see if we can update
            if goal_info is not None:

                # extract latest world_from_goal transform
                goal_pose = goal_info[0]

                # see if we should update the stored value
                if self.world_from_goal is None:
                    
                    # always update if there was no previously-seen goal
                    update_goal = True
                    
                else:
                    
                    # we already had a previously-seen goal, only update
                    # if this one is nearby
                    cur_goal_pos = self.world_from_goal.translation()
                    new_goal_pos = goal_pose.translation()

                    move_dist = np.linalg.norm(cur_goal_pos - new_goal_pos)
                    
                    if move_dist < POS_UPDATE_TOL:
                        update_goal = True

                # do goal update if needed
                if update_goal:
                    rospy.loginfo('updated goal: %s', goal_pose)
                    self.world_from_goal = goal_pose

        # check for new ball position
        if 'blue_ball' in self.latest_blobs:

            # find among detected objects
            b = self.find_closest_ball(cur_pose, self.latest_blobs['blue_ball'])

            # make sure ball not too close
            if b is not None:

                # see if we should update the stored value
                if self.ball_pos_world is None:

                    # always update if there was no previously-seen ball
                    update_ball = True

                elif np.linalg.norm(b - self.ball_pos_world) < POS_UPDATE_TOL:

                    # otherwise, update if current ball is nearby
                    # to previously seen one
                    update_ball = True

            # do update if needed
            if update_ball:
                rospy.loginfo('updated ball: %s', b)
                self.ball_pos_world = b

        # update kick position of either 
        if (self.world_from_goal is not None and
            self.ball_pos_world is not None and
            (update_goal or update_ball)):
            
            # calls your implementation to compute the kick position
            self.kick_pos_world = solution.compute_kick_pos_world(
                self.world_from_goal, self.ball_pos_world,
                BALL_SEP_DIST)

    ######################################################################
    # main control callback, called many times per second
    def control_callback(self, timer_event=None):

        # get current pose
        cur_pose = self.get_current_pose()

        # initialize zero command
        cmd_vel = Twist()

        if self.cliff_alert:

            # make sure stopped when lifted
            rospy.loginfo('stopped for safety...')
            
        elif not cur_pose:

            # make sure stopped when no pose yet
            rospy.loginfo('waiting for TransformListener...')
            
        else:

            # update the world model if needed
            if len(self.latest_blobs):
                self.update_world_model(cur_pose)
                self.latest_blobs = dict()
                
            if self.state == 'init':

                # transition from init to turn_to_kick when
                # kick position is ready
                if self.kick_pos_world is None:
                    rospy.loginfo('waiting for kick pos...')
                else:
                    self.state = 'turn_to_kick'

            elif self.state == 'turn_to_kick':

                # calls your implementation to do the "turn_towards" behavior
                a, done = solution.turn_towards(
                    cur_pose, self.kick_pos_world)

                # set up the command
                cmd_vel.angular.z = a

                # transition when done
                if done:
                    self.state = 'drive_to_kick'

            elif self.state == 'drive_to_kick':

                # calls your implementation to do the "drive_towards" behavior
                l, a, done = solution.drive_towards(
                    cur_pose, self.kick_pos_world)

                # set up the command
                cmd_vel.linear.x = l
                cmd_vel.angular.z = a

                rospy.loginfo('dist to kick pos: %f', np.linalg.norm(cur_pose.transform_inv(self.kick_pos_world)))

                # transition when done 
                if done:
                    self.state = 'turn_to_goal'

            elif self.state == 'turn_to_goal':

                # calls your implementation to do the "turn_towards" behavior
                a, done = solution.turn_towards(
                    cur_pose, self.world_from_goal.translation())

                # set up the command
                cmd_vel.angular.z = a

                # translation when done
                if done:
                    self.state = 'kick_it_whammo'


            elif self.state == 'kick_it_whammo':

                # finished when we have traveled
                dist = np.linalg.norm(self.kick_pos_world -
                                      cur_pose.translation())

                # slam on the gas
                cmd_vel.linear.x = 0.75
                
                if dist > KICK_DIST:
                    rospy.signal_shutdown('did we score a goal???')

        # truncate large commands for safety/stability
        if self.state != 'kick_it_whammo':
            
            cmd_vel.angular.z = clip('cmd_vel.angular.z',
                                     cmd_vel.angular.z,
                                     MAX_ANGULAR_Z)
            
            cmd_vel.linear.x = clip('cmd_vel.linear.x',
                                    cmd_vel.linear.x,
                                    MAX_LINEAR_X)

        # log the current state and command
        rospy.loginfo('in state %s, commanding linear=%f, angular=%f',
                      self.state, cmd_vel.linear.x, cmd_vel.angular.z)

        self.cmd_vel_pub.publish(cmd_vel)

    ######################################################################
    # called by main function below (after init)
    def run(self):
        
        # timers and callbacks are already set up, so just spin.
        # if spin returns we were interrupted by Ctrl+C or shutdown
        rospy.spin()

# main function
if __name__ == '__main__':
    try:
        ctrl = Controller()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass
    
        
