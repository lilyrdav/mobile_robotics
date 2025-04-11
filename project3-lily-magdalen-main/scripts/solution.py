import numpy as np
import rospy
from geometry_msgs.msg import Twist

# TODO: edit these constants to get smooth motion.
#
#   - values too low: robot will be sluggish/unresponsive;
#   - values too high: robot could skid and mess up odometry

ALPHA = 0.6 # Distance ahead on the gate x-axis for the target point

KP1 = 5.5
KP2 = 5

MAX_LINEAR_ACCEL = 10.0 # m/s^2
MAX_ANGULAR_ACCEL = 10.0 # rad/s^2

######################################################################
# filter a velocity to a maximum amount of acceleration
# NB: you should not need to edit this function!

def filter_vel(prev_vel, desired_vel, max_accel, dt):

    max_change = max_accel * dt

    new_min = prev_vel - max_change
    new_max = prev_vel + max_change

    return np.clip(desired_vel, new_min, new_max)

######################################################################
# calls function above to filter linear velocity
# NB: you should not need to edit this function!

def filter_linear_vel(prev_vel, desired_vel, dt):
    return filter_vel(prev_vel, desired_vel, MAX_LINEAR_ACCEL, dt)

######################################################################
# calls function above to filter angular velocity
# NB: you should not need to edit this function!

def filter_angular_vel(prev_vel, desired_vel, dt):
    return filter_vel(prev_vel, desired_vel, MAX_ANGULAR_ACCEL, dt)

######################################################################
# This function constitutes the "brains" of your slalom course
# implementation. The arguments are:
#
#   - cur_pose:   T_world_from_robot as a Transform2D() object
#
#   - cur_course: Current line in the course as an
#                 (action, lcolor, rcolor) tuple, for example
#                 ('left', 'green', 'green')
#
#   - cur_gate:   Either T_world_from_gate for the active gate
#                 the robot should be driving through, or None
#                 if no gate has been found yet.
#
#   - cur_tape:   Either a numpy array holding the (x, y) of
#                 the nearest tape blob in the world frame,
#                 or None if there was no tape blob detected.
#                 Note that your code should check to see if
#                 the tape blob is behind the robot and ignore
#                 it if so.
#
# This function should return a tuple (cmd_vel, cur_is_done):
#
#   - cmd_vel:    A geometry_msgs.msg.Twist holding the desired
#                 linear and angular velocity for the robot.
#                 You do not need to filter this velocity --
#                 the main controller will do this for you.
#
#  - cur_is_done: This should be True if the robot has passed
#                 through the current gate (e.g. the robot's x
#                 coordinate in the gate frame is positive).

def navigate_course(cur_pose,   # T_world_from_robot
                    cur_course, # current line in course 
                    cur_gate,   # T_world_from_gate or None
                    cur_tape):  # (x, y) of tape blob in world or None

    cmd_vel = Twist()
    cur_is_done = False

    cmd_vel.linear.x = 0.0
    cmd_vel.angular.z = 0.0

    if cur_gate is None:
        if cur_course[0] == 'left':
            cmd_vel.linear.x = 0.8
            cmd_vel.angular.z = 2.9
        elif cur_course[0] == 'right':
            cmd_vel.linear.x = 0.8
            cmd_vel.angular.z = -2.9
        elif cur_course[0] == 'tape' and cur_tape != None:
            new_position = cur_pose.inverse() * cur_tape
            new_pos_tx = new_position[0]
            new_pos_ty = new_position[1]
            error = np.arctan2(new_pos_ty, new_pos_tx)
            cmd_vel.linear.x = 0.7
            if new_pos_tx > 0.24:
                cmd_vel.angular.z = KP1 * error
            else:
                cmd_vel.angular.z = 0.0

        elif cur_tape == None or new_pos_tx < 0.0:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            rospy.loginfo("WARNING: no tape seen")
    
    else:
        T_world_from_robot = cur_pose
        T_world_from_gate = cur_gate

        T_gate_from_robot = T_world_from_gate.inverse() * T_world_from_robot
        T_robot_from_gate = T_gate_from_robot.inverse()

        dist_to_gate = np.linalg.norm(T_gate_from_robot.translation())
        #pure persuit 
        #compute p in the gate frame 
        p_gate_frame = T_gate_from_robot
        #p_c - the point in the gate frame with the same x-coordinate of p, 
        # but with a y-coordinate of zero.
        p_c = p_gate_frame.copy()
        p_c.y = 0.0
        #p_d - the point in the gate frame whose x-coordinate is alpha 
        # greater than the x-coordinate of p_c
        p_d = p_c.copy()
        p_d.x += ALPHA
        #map p_d from the gate frame to the robot frame
        p_d_robot_frame = T_robot_from_gate * p_d
        #compute angular error
        p_d_robot_x = p_d_robot_frame.x
        p_d_robot_y = p_d_robot_frame.y
        error = np.arctan2(p_d_robot_y, p_d_robot_x)
        cmd_vel.linear.x = 0.8
        if p_d_robot_x > 0.2:
            cmd_vel.angular.z = KP2 * error
        else:
            cmd_vel.angular.z = 0.0

        if p_gate_frame.x >= 0.0:
            cur_is_done = True

        rospy.loginfo('{}-{} gate is {} meters away from robot'.format(
            cur_course[1], cur_course[2], dist_to_gate))
            
    return cmd_vel, cur_is_done

######################################################################
