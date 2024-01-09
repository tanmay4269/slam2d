import numpy as np
from utils import *


def prediction_step(mu, sigma, odom, odom_cov, dt):

  theta = mu[2]
  v = odom[0]
  w = odom[1]
  
  mu[0] += -v/w * np.sin(theta) + v/w * np.sin(theta + w * dt)
  mu[1] += v/w * np.cos(theta) - v/w * np.cos(theta + w * dt)
  mu[2] += w * dt

  mu[2] = normalize_angle(mu[2])

  G = np.eye(mu.shape[0])
  G[0, 2] = -v/w * cos(theta) + v/w * cos(theta + w * dt)
  G[1, 2] = -v/w * sin(theta) + v/w * sin(theta + w * dt)

  sigma = G * sigma * G.T
  sigma[0:3, 0:3] += odom_cov


def correction_step(current_landmarks_ids, observed_landmarks):
  for idx in current_landmarks_ids:
    pass


def ekf():
  pass