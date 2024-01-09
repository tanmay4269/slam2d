import numpy as np

def normalize_angle(angle):
  normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi

  return normalized_angle


def deg2rad(deg):
  return deg * np.pi / 180.0

def max(a, b):
  if a > b:
    return a
  else:
    return b