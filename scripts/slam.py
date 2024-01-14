#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

import _config as config 

import utils
from utils import normalize_angle

from ekf import EKF
from iepf import iepf
from line_landmark import LandmarksDB

plt.rcParams["figure.figsize"] = config._plot_size

class SLAM(Node):
  def __init__(self):
    super().__init__("slam_nav")
    
    self.delta_time = 0
    self.time = self.get_clock().now().nanoseconds

    self.odom_listener = self.create_subscription(Odometry, 'odom', self.odom_callback, 100)  # last arg is queue size
    self.lidar_listener = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 100)

    """ Curr bot state """
    self.curr_odom = {
      "mean": np.zeros(2),  # linear, angular velocities 
      "cov" : np.zeros((3,3))
    }

    self.curr_pose = {
      "mean": np.zeros(3),  # x, y, yaw
      "cov" : np.zeros((3,3))
    }

    """ EKF """
    self.ekf = EKF()

    """ Landmarks """
    self.landmarks = LandmarksDB()
    self.true_pose = np.zeros(3)  # x, y, yaw

    """ Noise """
    self.lidar_cov = config._lidar_cov if config._use_fake_lidar_noise else np.zeros((2,2))


  def odom_callback(self, msg):
    if msg is not None:
      # I HAVE NO FUCKIN IDEA WHY TO USE 1E-8, WHILE ITS CLEARLY IN NANOSECONDS
      self.delta_time = (self.get_clock().now().nanoseconds - self.time) * 1e-8
      self.time = self.get_clock().now().nanoseconds

      vx = msg.twist.twist.linear.x
      vy = msg.twist.twist.linear.y
      w = msg.twist.twist.angular.z

      full_cov_mat = np.array(msg.twist.covariance).reshape(6,6)
      
      default_odom_cov = np.zeros((3, 3), dtype=float)
      default_odom_cov[:2, :2] = full_cov_mat[:2, :2]
      default_odom_cov[2, 2] = full_cov_mat[5, 5]

      self.curr_odom["cov"] = default_odom_cov

      if config._use_fake_odom_noise:
        self.curr_odom["cov"] += config._odom_cov
        vx, vy = np.random.multivariate_normal([vx, vy], self.curr_odom["cov"][:2, :2])
        w = np.random.normal(w, self.curr_odom["cov"][2, 2])

      linear = np.sqrt(vx ** 2 + vy ** 2)
      angular = w

      self.curr_odom["mean"] = [linear, angular]

      # true pose
      orientation = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
      ]

      self.true_pose[0] = msg.pose.pose.position.x
      self.true_pose[1] = msg.pose.pose.position.y
      self.true_pose[2] = euler_from_quaternion(orientation)[-1]


  def lidar_callback(self, msg):
    if msg is not None:
      readings = []
      
      ranges = np.array(msg.ranges)
      ranges = np.random.normal(ranges, np.sqrt(self.lidar_cov[0,0]))

      angles = msg.angle_min + np.arange(ranges.shape[0]) * msg.angle_increment
      angles = np.random.normal(angles, np.sqrt(self.lidar_cov[1,1]))
      
      angles = normalize_angle(angles)

      self.mapper(ranges, angles)
  

  def mapper(self, ranges, angles):
    """
    Runs EKF 
    """
    local_lines = None
    curr_landmark_ids = None
    
    # extract features when the bot isn't rotating 
    if np.abs(self.curr_odom["mean"][1]) < config._angular_vel_threshold:
      extracted_features = self.feature_extraction(ranges, angles)

      if extracted_features is not None:
          local_lines, curr_landmark_ids = extracted_features

    ekf_mu, ekf_sigma = self.ekf.run(
      self.curr_odom["mean"], self.curr_odom["cov"], self.delta_time,
      local_lines, curr_landmark_ids
    )

    self.curr_pose["mean"] = ekf_mu[:3]
    self.curr_pose["cov"]  = ekf_sigma[:3, :3]
    self.landmarks.update(ekf_mu[3:], curr_landmark_ids)

    self.show_landmarks()


  def feature_extraction(self, ranges, angles):
    """
    Range and angle are in the bot's local frame
    """

    # converting lider points from polar to euclidean
    lidar_pts = np.column_stack([
      self.curr_pose["mean"][0] + ranges * np.cos(angles),
      self.curr_pose["mean"][1] + ranges * np.sin(angles)
    ])

    # data cleaning
    lidar_pts = lidar_pts[
      ~np.any(np.isnan(lidar_pts) | \
      np.isinf(lidar_pts), axis=1)]

    if lidar_pts.shape[0] == 0:
      return

    """ DBSCAN """
    epsilon = config._dbscan_epsilon  # distance between points
    min_samples = config._dbscan_min_samples

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(lidar_pts)

    """ Plotting """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, color in zip(unique_labels, colors):
      if k == -1:
        color = [0, 0, 0, 1]

      class_member_mask = (labels == k)
      xy = lidar_pts[class_member_mask]
      plt.scatter(xy[:, 0], xy[:, 1], c=[color], s=config._dbscan_size)

    plt.draw()

    """ IEPF """
    local_lines = []
    curr_landmark_ids = []

    for k in unique_labels:
      if k == -1:
        continue

      lines, landmark_ids = iepf(
        lidar_pts[labels == k], self.curr_pose["mean"], self.landmarks)

      local_lines += lines
      curr_landmark_ids += landmark_ids

    local_lines = np.stack(local_lines, axis=0)

    curr_landmark_ids, unique_idx = np.unique(curr_landmark_ids, return_index=True)
    local_lines = local_lines[unique_idx]

    return (local_lines, curr_landmark_ids)

  # TODO: extend this for landmarks
  def plot_point(self, 
    mean, covariance_matrix=np.zeros((2,2)), 
    marker=None, markersize=10, color=None, label=None):

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    std_dev_x, std_dev_y = np.sqrt(eigenvalues)

    theta = np.linspace(0, 2 * np.pi, 10)
    ellipse_x = mean[0] + std_dev_x * np.cos(theta) * 10
    ellipse_y = mean[1] + std_dev_y * np.sin(theta) * 10

    ellipse = np.dot(np.array([ellipse_x, ellipse_y]).T, eigenvectors)

    plt.scatter(
      mean[0], mean[1], 
      color=color, marker=marker, s=markersize, label=label)
    plt.plot(ellipse[:, 0], ellipse[:, 1], color=color)


  def show_landmarks(self):
    plt.xlim(config._xlim)
    plt.ylim(config._ylim)

    self.plot_point(
      self.curr_pose["mean"][:2], self.curr_pose["cov"][:2, :2], 
      marker="o", color="red", label="Bot")
    
    self.plot_point(
      self.true_pose, 
      marker='*', markersize=100, color='blue', label='True Bot')

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