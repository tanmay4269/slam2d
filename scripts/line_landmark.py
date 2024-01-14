import numpy as np

import utils 
from utils import normalize_angle
import _config as config

class LandmarksDB:
  def __init__(self):
    self.lines = np.zeros((1, 3, 2))

    self.perp_dist_threshold = config._perp_dist_threshold
    self.linear_dist_threshold = config._linear_dist_threshold
    self.slope_threshold_rad = config._slope_threshold_rad
  
  def point_position(self, point, line):
    """
    -1 => point is outside to the left
     0 => point is inside
     1 => point is outside to the right
    """

    angle = np.arctan(line[0, 0])
    vec_l = point - (line[1] - np.array([self.linear_dist_threshold * np.cos(angle), self.linear_dist_threshold * np.sin(angle)]))
    vec_r = point - (line[2] + np.array([self.linear_dist_threshold * np.cos(angle), self.linear_dist_threshold * np.sin(angle)]))

    if np.dot(vec_l, vec_r.T) < 0:
      return 0
    elif np.linalg.norm(vec_l) < np.linalg.norm(vec_r):
      return -1
    else:
      return 1
  
  def new_line_to_add(self, query, line):
    lines_no_overlap_flag = -1
    far_line_flag = 0
    visited_landmark_flag = 1
    
    left_pt, left_dist = utils.orthogonal_projection(query[1], line[0])
    right_pt, right_dist = utils.orthogonal_projection(query[2], line[0])

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
    # endif

    return visited_landmark_flag

  def append(self, curr_pose, line):
    line = utils.local2global_line(curr_pose, line)

    angles = None
    curr_angle = None
    closest_slopes_idx = None

    angles = normalize_angle(np.arctan(self.lines[:, 0, 0]))  # can be improved by storing angles instead of slopes for lines 
    curr_angle = normalize_angle(np.arctan(line[0, 0]))

    closest_slopes_idx = np.where((angles >= curr_angle - self.slope_threshold_rad/2) & (angles <= curr_angle + self.slope_threshold_rad/2))[0]

    landmark_idx = -1

    if closest_slopes_idx.shape[0] == 0:
      landmark_idx = self.lines.shape[0]
      self.lines = np.insert(self.lines, landmark_idx, line, axis=0)

      return landmark_idx

    for idx in closest_slopes_idx:
      flag = self.new_line_to_add(line, self.lines[idx])

      if flag == 1:
        # this means the landmark was previously visited [for data association]
        landmark_idx = idx
        break

    if landmark_idx == -1 and closest_slopes_idx.shape[0] > 0:
      landmark_idx = self.lines.shape[0]
      self.lines = np.insert(self.lines, landmark_idx, line, axis=0)
      
    return landmark_idx

  def update(self, updated_polar_lines, curr_landmark_ids):
    """
    1. convert (rho, alpha) to (m, c)
    2. update old line
  --> [x] [2.1] project old line onto new 
      [ ] [2.2] first rotate to match m then translate to match c
    """

    if curr_landmark_ids is None:
      return

    # available_indices = curr_landmark_ids < self.lines.shape[0]
    # updated_polar_lines = updated_polar_lines.reshape(-1,2)[available_indices].reshape(-1)
    # curr_landmark_ids = curr_landmark_ids[available_indices]

    old_lines = self.lines[curr_landmark_ids]

    m, c = utils.polar2line(
      updated_polar_lines[::2], updated_polar_lines[1::2])

    coeffs = np.column_stack([m, c]).T

    # TODO: check if "points" arg is in correct shape
    projected_left_pts, _ = utils.orthogonal_projection(
      old_lines[:, 1].T,
      coeffs.T
    )

    projected_right_pts, _ = utils.orthogonal_projection(
      old_lines[:, 2].T,
      coeffs.T
    )

    print(coeffs.shape)
    print(projected_left_pts.shape)
    print(projected_right_pts.shape)

    self.lines[curr_landmark_ids] = np.stack([
      coeffs, projected_left_pts, projected_right_pts], axis=1)