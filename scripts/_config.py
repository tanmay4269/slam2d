import numpy as np

""" SLAM """
# artificial noise
_odom_cov = np.array([
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0]
])

_lidar_cov = np.array([
  [(1e-3)**2, 0.0],
  [0.0, (np.radians(0.1))**2]
])

""" DBSCAN """
_dbscan_epsilon = 0.2    # dist bw pts
_dbscan_min_samples = 3

""" RANSAC """
_deg_per_scan = 10         # angle around initial sample to form sub sample
_min_line_pts = 5          # stop algo if the sample has less than X number of nearest neighbours
_ransac_tolerence = 0.05   # if point is within x distance of line its part of line
_ransac_consensus = 10     # reject the line if less than X consensus are a part of it
_min_line_len = 0.8

""" IEPF """
_iepf_dist_theshold = 0.25
_iepf_length_threshold = 0.1

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