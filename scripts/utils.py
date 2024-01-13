import numpy as np

def normalize_angle(angle):
  normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi

  return normalized_angle

def local2global_point(curr_pose, point):
  r = np.linalg.norm(point)
  local_angle = np.arctan2(point[1], point[0])

  global_angle = normalize_angle(curr_pose[2] - local_angle)

  point[0] += r * np.cos(global_angle)
  point[1] += r * np.sin(global_angle)

def local2global_line(curr_pose, line):
  left_pt = local2global_point(line[1])
  right_pt = local2global_point(line[2])

  coeff = np.polyfit([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], 1)

  return np.array([
    coeff,
    left_pt,
    right_pt
  ])

def line2polar(line):
  """
  Returns orthogonal projection of the origin onto 
  the given line in polar coordinates
  """
  m, c = line[0]

  alpha = normalize_angle(np.arctan2(m) + np.pi/2)
  rho = np.abs(c * np.sin(alpha))

  return (rho, alpha)

def polar2line(rho, alpha):
  m = np.tan(alpha - np.pi/2)
  c = rho / np.sin(alpha)

  return (m, c)