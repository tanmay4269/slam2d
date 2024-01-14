import numpy as np

def zero_pad(array, padding):
  return np.pad(array, padding, mode="constant", constant_values=(0,0))

def normalize_angle(angle):
  normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi

  return normalized_angle

def orthogonal_projection(points, line_coeffs):
  """
  row0: x; row1: y

  points: (2, N)
  line_coeffs: (2, N)
  proj_pts: (N, 2)
  """

  print(f"points.shape = {points.shape}")
  print(f"line_coeffs.shape = {line_coeffs.shape}")

  x0 = (points[0] + line_coeffs[0] * points[1] - line_coeffs[0] * line_coeffs[1]) / (1 + line_coeffs[0]**2)
  y0 = line_coeffs[0] * x0 + line_coeffs[1]

  proj_pts = np.vstack([x0, y0]).T

  distances = np.abs(line_coeffs[0] * points[0] + line_coeffs[1] - points[1]) / np.sqrt(1 + line_coeffs[0]**2)

  return (proj_pts, distances)

def local2global_point(curr_pose, local_point):
  r = np.linalg.norm(local_point)
  local_angle = np.arctan2(local_point[1], local_point[0])

  global_angle = normalize_angle(curr_pose[2] - local_angle)

  global_pt = local_point.copy()

  global_pt[0] += r * np.cos(global_angle)
  global_pt[1] += r * np.sin(global_angle)

  return global_pt

def local2global_line(curr_pose, line):
  left_pt = local2global_point(curr_pose, line[1])
  right_pt = local2global_point(curr_pose, line[2])

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

  alpha = normalize_angle(np.arctan(m) + np.pi/2)
  rho = np.abs(c * np.sin(alpha))

  return (rho, alpha)

def polar2line(rho, alpha):
  m = np.tan(alpha - np.pi/2)
  c = rho / np.sin(alpha)

  return (m, c)