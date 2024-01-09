import numpy as np

from utils import *
import _config as config

class LandmarksDB:
  def __init__(self):
    self.lines = np.zeros((1, 3, 2))

    self.perp_dist_threshold = config._perp_dist_threshold
    self.linear_dist_threshold = config._linear_dist_threshold
    self.slope_threshold_rad = config._slope_threshold_rad

  
  def orthogonal_projection(self, point, line):
    line_coeff = line[0]

    x0 = (point[0] + line_coeff[0] * point[1] - line_coeff[0] * line_coeff[1]) / (1 + line_coeff[0]**2)
    y0 = line_coeff[0] * x0 + line_coeff[1]

    proj_pt = np.array([x0, y0])

    dist = np.abs(line_coeff[0] * point[0] + line_coeff[1] - point[1]) / np.sqrt(1 + line_coeff[0]**2)

    return (proj_pt, dist)

  
  def point_position(self, point, line):
    """
    -1 => point is outside to the left
     0 => point is inside
     1 => point is outside to the right
    """

    angle = np.arctan(line[0, 0])
    vec_l = point - (line[1] - np.array([self.linear_dist_threshold * np.cos(angle), self.linear_dist_threshold * np.sin(angle)]))
    vec_r = point - (line[2] + np.array([self.linear_dist_threshold * np.cos(angle), self.linear_dist_threshold * np.sin(angle)]))

    if np.dot(vec_l, vec_r) < 0:
      return 0
    elif np.linalg.norm(vec_l) < np.linalg.norm(vec_r):
      return -1
    else:
      return 1

  
  def new_line_to_add(self, query, line):

    lines_no_overlap_flag = -1
    far_line_flag = 0
    visited_landmark_flag = 1
    
    left_pt, left_dist = self.orthogonal_projection(query[1], line)
    right_pt, right_dist = self.orthogonal_projection(query[2], line)

    # bools to know if left and right points of query are within the line seg: `line`
    left_pt_position = self.point_position(left_pt, line)
    right_pt_position = self.point_position(right_pt, line)

    if max(left_dist, right_dist) >= self.perp_dist_threshold:
      return far_line_flag
    

    if left_pt_position == -1:

      if right_pt_position == -1:
        return lines_no_overlap_flag
      elif right_pt_position == 0:
        line[1] = query[1]
      else:
        line[1] = query[1]
        line[2] = query[2]
      
    elif left_pt_position == 0:
      
      if right_pt_position == -1:
        line[1] = query[2]
      elif right_pt_position == 1:
        line[2] = query[2]
    
    elif left_pt_position == 1:

      if right_pt_position == -1:
        line[1] = query[2]
        line[2] = query[1]
      elif right_pt_position == 0:
        line[2] = query[1]
      else:
        return lines_no_overlap_flag


    return visited_landmark_flag


  def append(self, line):

    angles = None
    curr_angle = None
    closest_slopes_idx = None

    angles = normalize_angle(np.arctan(self.lines[:, 0, 0]))
    curr_angle = normalize_angle(np.arctan(line[0, 0]))

    closest_slopes_idx = np.where((angles >= curr_angle - self.slope_threshold_rad/2) & (angles <= curr_angle + self.slope_threshold_rad/2))[0]

    if closest_slopes_idx.shape[0] == 0:
      self.lines = np.insert(self.lines, -1, line, axis=0)

    revisited_landmark_idx = -1

    for idx in closest_slopes_idx:
      flag = self.new_line_to_add(line, self.lines[idx])

      if flag == 1:
        # this means the landmark was previously visited [for data association]
        revisited_landmark_idx = idx
        break

    # zero count means none of the existing landmarks can be associated to the current query landmark
    if revisited_landmark_idx == -1 and closest_slopes_idx.shape[0] > 0:
      self.lines = np.insert(self.lines, -1, line, axis=0)

      revisited_landmark_idx = self.lines.shape[0]
      
    return revisited_landmark_idx