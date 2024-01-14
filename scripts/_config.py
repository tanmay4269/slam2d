import numpy as np

""" SLAM """
_use_fake_odom_noise = False
_odom_cov = np.array([
  [0.01, 0.0, 0.0],
  [0.0, 0.01, 0.0],
  [0.0, 0.0, 0.01]
])

_use_fake_lidar_noise = True
_lidar_cov = np.array([
  [(1e-3)**2, 0.0],
  [0.0, (np.radians(0.1))**2]
])

""" DBSCAN """
_dbscan_epsilon = 0.2  # dist bw pts
_dbscan_min_samples = 3

""" IEPF """
_iepf_dist_theshold = 0.25
_iepf_length_threshold = 0.1

""" EKF """
_Q_coeff = 1e-6

""" Landmarks """
_perp_dist_threshold = 0.5            # perp dist between point and line
_linear_dist_threshold = 0.05           # along the length of line seg
_slope_threshold_rad = np.radians(15)  # query get's compared with close enough sloped lines

_landmark_update_time = 0     # in ms
_angular_vel_threshold = 0.1  # this is to check if the bot isnt rotating, non rotating lidar is better in quality

""" Path """

""" Plotting """
_pause_time = 1e-6
_plot_size = (10, 10)
_xlim = (-15, 5)
_ylim = (-7, 15)

_dbscan_size = 10

_ransac_size = 8
_ransac_col = 'blue'

_landmark_size = 4
_landmark_col = 'green'