#!/usr/bin/env python3

# This python file runs a ROS 2-node of name waypoint_server which implements an action server to navigate the Swift Pico Drone to the given waypoints.
# You can use either PID or LQR controller to navigate the drone to the given waypoints.


import time
import math
from tf_transformations import euler_from_quaternion

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

#import control specific libraries
from waypoint_navigation.action import NavToWaypoint
#import the action

#pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from error_msg.msg import Error as PIDError
from controller_msg.msg import PIDTune 
from nav_msgs.msg import Odometry

class WayPointServer(Node): 
    '''
    Purpose:
    ---
    The WayPointServer class handles waypoint navigation for a drone using PID control.
    It processes various inputs, executes the PID control loop, and manages waypoint goals.
    '''

    def __init__(self):  
        '''
        Purpose:
        ---
        Initializes the WayPointServer node and sets up subscribers, publishers, timers,
        and the action server for handling navigation goals.

        Example Call:
        ---
        waypoint_server = WayPointServer()

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''

        super().__init__('waypoint_server')

        # Initialize callback groups for thread safety
        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        # Variables to track the drone's state
        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0
        self.drone_position = [0.0, 0.0, 0.0, 0.0]
        self.setpoint = [0, 0, 27, 0] 
        self.dtime = 0
        self.initial_yaw = None

        # Initialize command message for drone control
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # PID controller parameters
        self.Kp = [20, 20, 30, 15]
        self.Ki = [0.15, 0.15, 8, 0.05]
        self.Kd = [40, 40, 40, 5]
        self.integral_limit = [30.0, 30.0, 20.0, 20.0]
        self.alpha = 0.95

        # Error tracking for PID control
        self.error = [0, 0, 0, 0]
        self.prev_error = [0, 0, 0, 0]
        self.diff_error = [0, 0, 0, 0]
        self.sum_error = [0, 0, 0, 0]
        self.prev_diff_error = [0.0, 0.0, 0.0, 0.0]

        # PID loop time step
        self.sample_time = 0.010

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)
        self.pid_error = PIDError()

        # Subscribers
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)
        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10)
        
        # Action server for waypoint navigation
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            self.execute_callback,
            callback_group=self.action_callback_group
        )        
        
        # Initialize the drone (arm it)
        self.arm()

        # Timer for the PID loop
        self.timer = self.create_timer(self.sample_time, self.pid, callback_group=self.pid_callback_group)


    def disarm(self):
        '''
        Purpose:
        ---
        Disarms the drone by setting all control channels to their minimum values.

        Example Call:
        ---
        waypoint_server.disarm()

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''

        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)


    def arm(self):
        '''
        Purpose:
        ---
        Arms the drone by setting all control channels to neutral positions.

        Example Call:
        ---
        waypoint_server.arm()

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''

        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)


    def whycon_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the drone's position based on the PoseArray message received from the WhyCon system.

        Example Call:
        ---
        waypoint_server.whycon_callback(pose_array_message)

        Input Arguments:
        ---
        `msg` : [PoseArray]
            PoseArray message containing the current position of the drone.

        Returns:
        ---
        None
        '''

        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z 

        self.dtime = msg.header.stamp.sec


    def altitude_set_pid(self, alt):
        '''
        Purpose:
        ---
        Updates the PID gains for altitude based on the PIDTune message received.

        Example Call:
        ---
        waypoint_server.altitude_set_pid(pid_tune_message)

        Input Arguments:
        ---
        `alt` : [PIDTune]
            PIDTune message containing the new PID parameters for altitude control.

        Returns:
        ---
        None
        '''

        self.Kp[2] = alt.kp * 0.03
        self.Ki[2] = alt.ki * 0.008
        self.Kd[2] = alt.kd * 0.6


    def pitch_set_pid(self, pitch):
        '''
        Purpose:
        ---
        Updates the PID gains for pitch based on the PIDTune message received.

        Example Call:
        ---
        waypoint_server.pitch_set_pid(pid_tune_message)

        Input Arguments:
        ---
        `pitch` : [PIDTune]
            PIDTune message containing the new PID parameters for pitch control.

        Returns:
        ---
        None
        '''

        self.Kp[1] = pitch.kp * 0.03
        self.Ki[1] = pitch.ki * 0.008
        self.Kd[1] = pitch.kd * 0.6


    def roll_set_pid(self, roll):
        '''
        Purpose:
        ---
        Updates the PID gains for roll based on the PIDTune message received.

        Example Call:
        ---
        waypoint_server.roll_set_pid(pid_tune_message)

        Input Arguments:
        ---
        `roll` : [PIDTune]
            PIDTune message containing the new PID parameters for roll control.

        Returns:
        ---
        None
        '''

        self.Kp[0] = roll.kp * 0.03
        self.Ki[0] = roll.ki * 0.008
        self.Kd[0] = roll.kd * 0.6


    def odometry_callback(self, msg):
        '''
        Purpose:
        ---
        Updates the drone's orientation based on the Odometry message received.

        Example Call:
        ---
        waypoint_server.odometry_callback(odometry_message)

        Input Arguments:
        ---
        `msg` : [Odometry]
            Odometry message containing the current orientation of the drone.

        Returns:
        ---
        None
        '''

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        self.roll_deg = math.degrees(roll)
        self.pitch_deg = math.degrees(pitch)
        self.yaw_deg = math.degrees(yaw)
        self.drone_position[3] = (self.yaw_deg)

        if self.initial_yaw is None:
            # Yaw setpoint is the initial yaw angle
            self.initial_yaw = self.yaw_deg
            self.setpoint[3] = self.yaw_deg


    def pid(self):
        '''
        Purpose:
        ---
        Executes the PID control loop to compute control commands for the drone.

        Example Call:
        ---
        waypoint_server.pid()

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''

        pid_output = [0.0, 0.0, 0.0, 0.0]

        # Calculate errors for each axis
        for i in range(3):
            self.error[i] = self.drone_position[i] - self.setpoint[i]
            
            self.sum_error[i] += self.error[i] * self.sample_time
            self.sum_error[i] = max(min(self.sum_error[i], self.integral_limit[i]), -self.integral_limit[i])
            
            self.diff_error[i] = (self.error[i] - self.prev_error[i]) / self.sample_time
            self.diff_error[i] = self.alpha * self.prev_diff_error[i] + (1 - self.alpha) * self.diff_error[i]
            self.prev_diff_error[i] = self.diff_error[i]
            
            # PID equation
            pid_output[i] = self.Kp[i] * self.error[i] + self.Ki[i] * self.sum_error[i] + self.Kd[i] * self.diff_error[i]

            self.prev_error[i] = self.error[i]


        if self.initial_yaw is not None:
            self.error[3] = self.drone_position[3] - self.initial_yaw
            self.sum_error[3] += self.error[3] * self.sample_time
            self.diff_error[3] = (self.error[3] - self.prev_error[3]) / self.sample_time
            pid_output[3] = self.Kp[3] * self.error[3] + self.Ki[3] * self.sum_error[3] + self.Kd[3] * self.diff_error[3]
            
            self.prev_error[3] = self.error[3]
            self.pid_error.yaw_error = self.error[3]

            self.cmd.rc_yaw = int(1500 + pid_output[3])
            self.cmd.rc_yaw = max(1000, min(2000, self.cmd.rc_yaw))

        self.pid_error.roll_error = self.error[0]
        self.pid_error.pitch_error = self.error[1]
        self.pid_error.throttle_error = self.error[2]

        self.cmd.rc_roll = int(1500 - pid_output[0])
        self.cmd.rc_roll = max(1000, min(2000, self.cmd.rc_roll))
        
        self.cmd.rc_pitch = int(1500 + pid_output[1])
        self.cmd.rc_pitch = max(1000, min(2000, self.cmd.rc_pitch))

        self.cmd.rc_throttle = int(1500 + pid_output[2])
        self.cmd.rc_throttle = max(1000, min(2000, self.cmd.rc_throttle))


        self.command_pub.publish(self.cmd)
        self.pid_error_pub.publish(self.pid_error)


    def execute_callback(self, goal_handle):
        '''
        Purpose:
        ---
        Handles the execution of a waypoint navigation goal by setting the setpoint for the drone.

        Example Call:
        ---
        result = waypoint_server.execute_callback(goal_handle)

        Input Arguments:
        ---
        `goal_handle` : [GoalHandle]
            The goal handle object containing the waypoint target.

        Returns:
        ---
        `result` : [NavToWaypoint.Result]
            The result of the navigation action indicating success.
        '''

        self.get_logger().info('Executing goal...')
        self.setpoint[0] = goal_handle.request.waypoint.position.x
        self.setpoint[1] = goal_handle.request.waypoint.position.y
        self.setpoint[2] = goal_handle.request.waypoint.position.z
        self.setpoint[3] = self.initial_yaw
        self.get_logger().info(f'New Waypoint Set: {self.setpoint}')
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime

        feedback_msg = NavToWaypoint.Feedback()
        result = NavToWaypoint.Result()
        
        while True:
            feedback_msg.current_waypoint.pose.position.x = self.drone_position[0]
            feedback_msg.current_waypoint.pose.position.y = self.drone_position[1]
            feedback_msg.current_waypoint.pose.position.z = self.drone_position[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            goal_handle.publish_feedback(feedback_msg)

            drone_is_in_sphere = self.is_drone_in_sphere(self.drone_position, goal_handle, 0.4)
            
            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        pass
            
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        self.point_in_sphere_start_time = self.dtime

            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                        self.get_logger().info('Drone in sphere')
                             
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.get_logger().info('Drone out of sphere')
                        self.point_in_sphere_start_time = None

            if self.time_inside_sphere > self.max_time_inside_sphere:
                 self.max_time_inside_sphere = self.time_inside_sphere

            if self.max_time_inside_sphere >= 3:
                 break                        

        # Once the waypoint is reached, return success
        goal_handle.succeed()

        result.hov_time = self.dtime - self.duration
        return result


    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        '''
        Purpose:
        ---
        Checks if the drone is within a spherical region centered around the setpoint.

        Example Call:
        ---
        is_in_sphere = waypoint_server.is_drone_in_sphere()

        Input Arguments:
        ---
        None

        Returns:
        ---
        `is_in_sphere` : [bool]
            True if the drone is within the spherical region, False otherwise.
        '''

        return (
            (drone_pos[0] - sphere_center.request.waypoint.position.x) ** 2
            + (drone_pos[1] - sphere_center.request.waypoint.position.y) ** 2
            + (drone_pos[2] - sphere_center.request.waypoint.position.z) ** 2
        ) <= radius**2


def main(args=None):
    '''
    Purpose:
    ---
    Initializes the WayPointServer node and starts the ROS2 multi-threaded executor.

    Example Call:
    ---
    main()

    Input Arguments:
    ---
    `args` : [list], optional
        Arguments for the ROS2 node.

    Returns:
    ---
    None
    '''

    rclpy.init(args=args)

    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)
    
    try:
         executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
         waypoint_server.destroy_node()
         rclpy.shutdown()


if __name__ == '__main__':
    main()
