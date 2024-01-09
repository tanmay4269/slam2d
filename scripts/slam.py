#! /usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

from utils import *
import _config as config 
from ransac import *
from iepf import *
from ekf import ekf

plt.rcParams["figure.figsize"] = config._plot_size

class SLAM(Node):
  def __init__(self):
    super().__init__("slam_nav")
    
    self.time = self.get_clock().now().nanoseconds
    self.start_time = self.time
    self.delta_time = 0

    self.odom_listener = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)  # queue_size of 10
    self.lidar_listener = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)

    ''' Curr bot state '''  # replace contents in this with those in "EKF"
    self.curr_odom = []
    self.curr_pose = [0.0, 0.0, 0.0]

    ''' EKF '''
    self.mean_state = np.zeros((3, 1))
    self.covariance_matrix = np.zeros((3, 3))

    ''' Landmarks '''
    self.landmark_update_time = config._landmark_update_time
    self.landmarks = LandmarksDB()
    self.raw_lidar_data = []
    self.angular_vel_threshold = config._angular_vel_threshold

    ''' Noise '''
    # x, y, yaw
    self.odom_cov = config._odom_cov
    
    # range, angle
    self.lidar_cov = config._lidar_cov

    ''' Path '''
    self.true_trajectory      = {'x': [0.0], 'y': [0.0], 'yaw': [0.0]}
    self.pure_odom_trajectory = {'x': [0.0], 'y': [0.0], 'yaw': [0.0]}
    self.slam_trajectory      = {'x': [], 'y': [], 'yaw': []}


  def lidar_callback(self, msg):
    if msg is not None:
      readings = []
      
      ranges = np.array(msg.ranges)
      ranges = np.random.normal(ranges, np.sqrt(self.lidar_cov[0,0]))

      angles = msg.angle_min + np.arange(ranges.shape[0]) * msg.angle_increment
      angles = normalize_angle(angles + np.pi - self.true_trajectory["yaw"][-1])

      angles = np.random.normal(angles, np.sqrt(self.lidar_cov[1,1]))

      readings = np.column_stack([
          self.true_trajectory["x"][-1] + ranges * np.cos(angles),
          self.true_trajectory["y"][-1] + ranges * np.sin(angles)
      ]).T

      # readings = np.column_stack([
      #     self.curr_pose[0] + msg.ranges * np.cos(angles),
      #     self.curr_pose[1] + msg.ranges * np.sin(angles)
      # ])

      # reverting angles else it'll mess up other systems
      angles = normalize_angle(angles - np.pi + self.true_trajectory["yaw"][-1])

      self.raw_lidar_data = np.vstack((angles, readings))

      self.show_landmarks()

      print(f"Landmark Count: {self.landmarks.lines.shape[0]}", end='\r', flush=True)

      current_landmarks_ids = []

      # if int((self.time - self.start_time) * 1e-6) > self.landmark_update_time:
      #   self.start_time = self.time

      if np.abs(self.curr_odom[1]) < self.angular_vel_threshold:
        current_landmarks_ids = self.feature_extraction(self.raw_lidar_data)

        ekf()


  def odom_callback(self, msg):
    if msg is not None:

      # getting simulation time
      self.delta_time = self.get_clock().now().nanoseconds - self.time
      self.time = self.get_clock().now().nanoseconds

      # estimating trajectory with odometry alone
      self.predict_pose_pure_odom(msg)

      # true trajectory
      self.get_true_pose(msg)


  def predict_pose_pure_odom(self, msg):
    
    # perfect odom
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    w = msg.twist.twist.angular.z

    # adding artificial noise 
    # vx, vy, w = np.random.multivariate_normal(
    #   mean = [vx, vy, w],
    #   cov  = self.odom_cov
    # )

    # noisy odom
    self.curr_odom = [
      np.hypot(vx, vy),  # linear speed
      w
    ]

    self.curr_pose = self.motion_model(
      # prev pose 
      [
        self.pure_odom_trajectory['x'][-1],
        self.pure_odom_trajectory['y'][-1],
        self.pure_odom_trajectory['yaw'][-1]
      ],
      self.curr_odom,
      self.delta_time * 1e-9
    )

    self.pure_odom_trajectory['x'].append(self.curr_pose[0])
    self.pure_odom_trajectory['y'].append(self.curr_pose[1])
    self.pure_odom_trajectory['yaw'].append(self.curr_pose[2])


  # p(x' | x, u)
  def motion_model(self, prev_pose, curr_odom, dt):
    x = prev_pose[0]
    y = prev_pose[1]
    yaw = prev_pose[2]

    v = curr_odom[0]
    w = curr_odom[1]

    x += -v/w * np.sin(yaw) + v/w * np.sin(yaw + w * dt)
    y += v/w * np.cos(yaw) - v/w * np.cos(yaw + w * dt)
    yaw += w * dt 

    yaw = normalize_angle(yaw)

    return (x, y, yaw)


  def get_true_pose(self, msg):
    pose = msg.pose.pose
    orientation = [
      pose.orientation.w, 
      pose.orientation.x, 
      pose.orientation.y, 
      pose.orientation.z
    ]

    current_pose = [
      pose.position.x, 
      pose.position.y, 
      euler_from_quaternion(orientation)[0]
    ]
    
    self.true_trajectory['x'].append(current_pose[0])
    self.true_trajectory['y'].append(current_pose[1])
    self.true_trajectory['yaw'].append(current_pose[2])

  
  def feature_extraction(self, data):
    
    # data cleaning
    data = data[:, ~np.any(np.isnan(data) | np.isinf(data), axis=0)]
    
    dbscan_data = data[1:, :].T

    if dbscan_data.shape[0] == 0:
      return

    epsilon = config._dbscan_epsilon  # distance between points
    min_samples = config._dbscan_min_samples

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(dbscan_data) 

    # Plot the data and cluster assignments
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Outliers are marked in black
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = dbscan_data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=config._dbscan_size)

    plt.draw() 

    current_landmarks_ids = []

    for k in unique_labels:
      if k == -1:
        continue

      # current_landmarks_ids += ransac(data[:, labels==k], self.landmarks)
      current_landmarks_ids += iepf(data[:, labels==k], self.landmarks)

    return current_landmarks_ids


  ''' Rendering stuff '''

  def show_landmarks(self):
    ### Plotting is super expensive
    # plt.scatter(self.true_trajectory['x'], self.true_trajectory['y'], s=8, label='True Trajectory')
    # plt.scatter(self.pure_odom_trajectory['x'], self.pure_odom_trajectory['y'], s=8, label='Pure Odom Trajectory')

    plt.xlim(config._xlim)
    plt.ylim(config._ylim)

    # plt.plot(self.curr_pose[0], self.curr_pose[1], marker='o', markersize=8, color='Red', label='Bot')
    plt.plot(self.true_trajectory['x'][-1], self.true_trajectory['y'][-1], marker='o', markersize=8, color='Red', label='Bot')

    plt.plot(
      [self.landmarks.lines[:, 1, 0], self.landmarks.lines[:, 2, 0]], 
      [self.landmarks.lines[:, 1, 1], self.landmarks.lines[:, 2, 1]],
      marker = 'o',
      color = config._landmark_col,
      markersize = config._landmark_size
    )

    plt.legend()
    plt.draw()
    plt.pause(config._pause_time)
    plt.clf()

def main():
  rclpy.init()
  
  main = SLAM()

  try:
    rclpy.spin(main)
  except KeyboardInterrupt:
    print("Shutting down...")

  main.destroy_node()  
  rclpy.shutdown()

if __name__ == "__main__":
  main()