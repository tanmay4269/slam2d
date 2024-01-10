import numpy as np

def normalize_angle(angle):
  normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi

  return normalized_angle

def line2polar(line):
  m, c = line[0]

  alpha = normalize_angle(np.arctan2(m) + np.pi/2)
  rho = np.abs(c) * sin(alpha)

  return (rho, alpha)
  