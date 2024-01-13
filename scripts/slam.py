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
import iepf
from ekf import EKF

plt.rcParams["figure.figsize"] = config._plot_size

class SLAM(Node):
  def __init__(self):
    super().__init__("slam_nav")
    
    self.time = self.get_clock().now().nanoseconds
    self.delta_time = 0

    self.odom_listener = self.create_subscription(Odometry, 'odom', self.odom_callback, 100)  # last arg is queue size
    self.lidar_listener = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 100)

    """ Curr bot state """
    self.curr_odom = [0.0, 0.0]       # linear, angular velocities 
    self.curr_pose = [0.0, 0.0, 0.0]  # x, y, yaw

    """ EKF """
    self.ekf = EKF()

    """ Landmarks """
    self.landmarks = LandmarksDB()

    """ Noise """
    self.odom_cov  = None if not config._use_fake_odom_noise else config._odom_cov
    self.lidar_cov = np.zeros((2,2)) if not config._use_fake_lidar_noise else config._lidar_cov


  def odom_callback(self, msg):
    if msg is not None:
      # getting simulation time
      self.delta_time = self.get_clock().now().nanoseconds - self.time
      self.time = self.get_clock().now().nanoseconds

      if not config._use_fake_odom_noise:
        c = np.array(msg.twist.covariance).reshape(6,6)
        self.odom_cov = np.array([
          [c[0, :2], 0.0],
          [c[1, :2], 0.0],
          [0.0, 0.0, c[5,5]]
        ])

      vx = msg.twist.twist.linear.x
      vy = msg.twist.twist.linear.y
      w = msg.twist.twist.angular.z

      vx, vy = np.random.multivariate_normal([vx, vy], self.odom_cov[:2, :2])
      w = np.random.normal(w, self.odom_cov[2, 2])

      linear = np.sqrt(vx ** 2 + vy ** 2)
      angular = w

      self.curr_odom = [linear, angular]


  def lidar_callback(self, msg):
    if msg is not None:
      readings = []
      
      ranges = np.array(msg.ranges)
      ranges = np.random.normal(ranges, np.sqrt(self.lidar_cov[0,0]))

      angles = msg.angle_min + np.arange(ranges.shape[0]) * msg.angle_increment
      angles = np.random.normal(angles, np.sqrt(self.lidar_cov[1,1]))
      
      angles = normalize_angle(angles)

      local_lines = None
      curr_landmark_ids = None

      # extract features when the bot isn't rotating 
      if np.abs(self.curr_odom[1]) < config._angular_vel_threshold:
        local_lines, curr_landmark_ids = self.feature_extraction(ranges, angles)

        ekf_mu, ekf_sigma = self.ekf.run(
          self.curr_odom, self.odom_cov, self.delta_time,
          local_lines, curr_landmark_ids
        )

        self.curr_pose = ekf_mu[:3]
        self.landmarks.update(ekf_mu[3:], curr_landmark_ids)


  def feature_extraction(self, ranges, angles):
    """
    Range and angle are in the bot's local frame
    """

    # converting lider points from polar to euclidean
    lidar_pts = np.column_stack([
      self.curr_pose[0] + ranges * np.cos(angles),
      self.curr_pose[1] + ranges * np.sin(angles)
    ])
    
    # data cleaning
    lidar_pts = lidar_pts[:, ~np.any(np.isnan(lidar_pts) | np.isinf(lidar_pts), axis=0)]
    
    """ DBSCAN """
    dbscan_data = lidar_pts[1:, :].T

    if dbscan_data.shape[0] == 0:
      return

    epsilon = config._dbscan_epsilon  # distance between points
    min_samples = config._dbscan_min_samples

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(dbscan_data) 

    # Plotting
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, color in zip(unique_labels, colors):
      if k == -1:
        color = [0, 0, 0, 1]

      class_member_mask = (labels == k)
      xy = dbscan_data[class_member_mask]
      plt.scatter(xy[:, 0], xy[:, 1], c=[color], s=config._dbscan_size)

    plt.draw()

    """ IEPF """
    local_lines = []
    curr_landmark_ids = []

    for k in unique_labels:
      if k == -1:
        continue

      lines, landmark_ids = iepf(
        lidar_pts[:, labels==k], 
        self.curr_pose, self.landmarks)

      local_lines += lines
      curr_landmark_ids += landmark_ids

    return (local_lines, curr_landmark_ids)


  def show_landmarks(self):
    plt.xlim(config._xlim)
    plt.ylim(config._ylim)

    plt.plot(
      self.curr_pose[0], self.curr_pose[1], 
      marker='o', markersize=8, color='Red', label='Bot')

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