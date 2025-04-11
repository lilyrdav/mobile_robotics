import roslib; roslib.load_manifest('project2')
import rospy
import rospkg

from transform2d import Transform2D
import numpy as np

######################################################################
#
# Computes the target position of the robot to line up to kick the ball.
#
# Arguments:
#
#   - world_from_goal: pose of goal in world frame expressed as
#                      Transform2D to world from goal frame
#
#   - ball_pos_world:  position of ball in world frame as a numpy
#                      array of length 2
#
# Returns the target position of the robot in the world frame.

def compute_kick_pos_world(world_from_goal, ball_pos_world, sep_dist):
    # Get the goal position in the world frame
    goal_pos_world = world_from_goal.translation()

    # Calculate the vector from the goal to the ball
    goal_to_ball = ball_pos_world - goal_pos_world
    goal_to_ball_norm = np.linalg.norm(goal_to_ball)

    if goal_to_ball_norm == 0:
        # Avoid division by zero if the ball is somehow at the same position as the goal
        raise ValueError("Ball is at the goal position, cannot determine direction.")

    # Normalize the vector
    goal_to_ball_unit = goal_to_ball / goal_to_ball_norm

    # Calculate the kick position which is sep_dist away from the ball, on the line extending from the goal through the ball
    kick_pos_world = ball_pos_world + goal_to_ball_unit * sep_dist
    kick_pos_world[0] = kick_pos_world[0] + 0.1
    kick_pos_world[1] = kick_pos_world[1] + 0.1
    #**when the robot is coming from the right towards the ball we need to add 0.1

    return kick_pos_world

######################################################################
#
# Behavior for turning towards a target point.
#
# Arguments:
#
#  - world_from_robot: pose of robot in world frame expressed as
#                      Transform2D to world from robot frame
#
#  - target_pos_world: target point coordinates in world frame as a
#                      numpy array of length 2
#
# Returns a tuple of two values:
#
#  - a:    desired angular velocity of the robot
#
#  - done: True when the robot is sufficiently close to facing the
#          goal point, False otherwise

def turn_towards(world_from_robot, target_pos_world):
    kp_theta = 1.0  # This value should be tuned for your specific robot
    kb_theta = 0.1  # This value should be small to prevent slow convergence

    target_pos_robot = world_from_robot.transform_inv(target_pos_world)
    
    epsilon_theta = np.arctan2(target_pos_robot[1], target_pos_robot[0])
    
    a = kp_theta * epsilon_theta + kb_theta * np.sign(epsilon_theta)
    
    angle_threshold = np.radians(2)  # 2 degrees threshold
    done = np.abs(epsilon_theta) < angle_threshold
    
    return a, done

######################################################################
#
# Behavior for driving towards a target point.
#
# Arguments:
#
#  - world_from_robot: pose of robot in world frame expressed as
#                      Transform2D to world from robot frame
#
#  - target_pos_world: target point coordinates in world frame as a
#                      numpy array of length 2
#
# Returns a tuple of three values:
#
#  - l:    desired linear velocity of the robot
#
#  - a:    desired angular velocity of the robot
#
#  - done: True when the robot is sufficiently close to facing the
#          goal point, False otherwise


def drive_towards(world_from_robot, target_pos_world):

    l = 0.0

    a = 0.0

    done = False
    
    kp_x = 0.7
    kb_x = 0.1
    #calculate error 
    target_pos_robot = world_from_robot.transform_inv(target_pos_world)

    error = target_pos_robot[0]

    #get the linear velocity 
    l = (kp_x*error) + kb_x

    #get the angular velocity
    kp_theta = 0.1

    epsilon_theta = np.arctan2(target_pos_robot[1], target_pos_robot[0])
    
    #steering is disabled when the distance is less than 0.25
    if epsilon_theta > 0.25:
        
        a = kp_theta * epsilon_theta + np.sign(epsilon_theta)

    # stop when the ball is some distance away 
    if error < 0.4:
        done = True
    
    return l, a, done


######################################################################
#
# Helper function to test your compute_kick_pos_world implementation.
# You should not modify this function.

def _test_kick_pos():

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def draw_circle(pos, radius, label=None, color=None):
        plt.gca().add_patch(Ellipse(pos, 2*radius, 2*radius, fc=color))
        if label is not None:
            plt.text(pos[0] + 1.2*radius, pos[1], label)

    g = [4.0, 1.0]
            
    world_from_goal = Transform2D(g[0], g[1], 0.5)

    pl = world_from_goal * [0, 0.5]
    pr = world_from_goal * [0, -0.5]

    b = world_from_goal * [-0.9, 0.2]

    k = compute_kick_pos_world(world_from_goal, b, 0.4)

    draw_circle(pl, 0.05, 'left cone', 'g')
    draw_circle(pr, 0.05, 'right cone', 'g')

    draw_circle(g, 0.03, 'goal center', [0.8, 0.8, 0.8])

    draw_circle(b, 0.15, 'ball', 'b')

    draw_circle(k, 0.03, 'kick pos', 'm')

    plt.plot([b[0], k[0]], [b[1], k[1]], 'm--')
    plt.plot([b[0], g[0]], [b[1], g[1]], 'y--')

    plt.axis('equal')
    plt.show()

    
if __name__ == '__main__':
    _test_kick_pos()
