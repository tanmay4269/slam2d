"""
Based off of paper: https://www.joace.org/uploadfile/2014/0113/20140113054354731.pdf
and shares concepts from slam course: http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam05-ekf-slam.pdf
"""

import numpy as np
from numpy import sin, cos

import utils
from utils import normalize_angle

import _config as config

class EKF:
  def __init__(self):
    # global (x_bot, y_bot, yaw_bot, (rho, alpha) for all observed landmarks) 
    self.mu = np.zeros((3, 1))
    self.sigma = np.zeros((3, 3))
    self.observed_landmarks = None

  def run(self,
    curr_odom, odom_cov, dt,
    local_lines, curr_landmark_ids):

    self.prediction_step(curr_odom, odom_cov, dt)
    self.correction_step(local_lines, curr_landmark_ids)

    return (self.mu, self.sigma)
    
  def prediction_step(self, odom, odom_cov, dt):
    """
    This step exclusively predicts bot's pose, it's mean and covariance
    - self.mu: pose at t-1
    - odom: odom at t
    """

    v = odom[0]
    w = odom[1]
    prev_theta = self.mu[2]
    
    self.mu[0] += -v/w * sin(prev_theta) + v/w * sin(prev_theta + w * dt)
    self.mu[1] +=  v/w * cos(prev_theta) - v/w * cos(prev_theta + w * dt)
    self.mu[2] += w * dt

    self.mu[2] = normalize_angle(self.mu[2])

    G = np.eye(self.mu.shape[0])
    G[0, 2] = -v/w * cos(prev_theta) + v/w * cos(prev_theta + w * dt)
    G[1, 2] = -v/w * sin(prev_theta) + v/w * sin(prev_theta + w * dt)

    self.sigma = G * self.sigma * G.T
    self.sigma[:3, :3] += odom_cov

  def correction_step(self, lines, curr_landmark_ids):
    """
    mu: x, y, theta of bot and (rho, alpha) for all landmarks in global frame
    lines: in local frame
    """

    """ 
    zero padding:
      - observed_landmarks 
      - mu, sigma
    """
    if curr_landmark_ids is None:
      return

    i_max = np.max(curr_landmark_ids) + 1
    pad = 0
    
    if self.observed_landmarks is None:
      pad = i_max 
      self.observed_landmarks = []
    else:
      pad = max(0, i_max - self.observed_landmarks.shape[0])

    self.observed_landmarks = utils.zero_pad(self.observed_landmarks, (0, pad))
    
    N_old = self.mu.shape[0]
    self.mu = utils.zero_pad(self.mu, ((0, 2 * pad), (0, 0)))
    self.sigma = utils.zero_pad(self.sigma, ((0, 2 * pad), (0, 2 * pad)))

    N_new = self.mu.shape[0]
    for i in range(N_old, N_new):
      self.sigma[i,i] = np.inf 

    # Z === readings at time = t
    # Z is (rho, alpha) in "LOCAL FRAME"
    # expected_Z is predicted Z from info from t-1 and pediction_step
    num_lines = lines.shape[0]
    Z = np.zeros((2 * num_lines, 1))
    expected_Z = np.zeros((2 * num_lines, 1))

    # stacked Jacobian blocks 
    # will be of shape: (2m, 3 + 2N), N being total number of landmarks
    H = None

    for idx in curr_landmark_ids:
      # for new landmarks
      if self.observed_landmarks[idx] == 0:
        rho, alpha = utils.line2polar(lines[idx])

        self.mu[2 * idx + 3] = rho + \
          self.mu[0] * cos(alpha) + \
          self.mu[1] * sin(alpha)

        self.mu[2 * idx + 4] = normalize_angle(self.mu[2] + alpha)

        self.observed_landmarks[idx] = 1
      
      rho, alpha = utils.line2polar(lines[idx])  # in local frame

      Z[2 * idx] = rho
      Z[2 * idx + 1] = alpha

      expected_Z[2 * idx] = self.mu[2 * idx] - \
        (self.mu[0] * cos(alpha) + self.mu[1] * sin(alpha))

      expected_Z[2 * idx + 1] = normalize_angle(self.mu[2 * idx + 1] - self.mu[2])

      Hi = np.array([
        [-cos(alpha), -sin(alpha), 0, 1, (self.mu[0] * sin(alpha) - self.mu[1] * cos(alpha))],
        [0, 0, -1, 0, 1]
      ], dtype=float)

      Fx = np.zeros((5, N_new))
      Fx[:3, :3] = np.eye(3)
      Fx[3, 2 * idx + 3] = 1
      Fx[4, 2 * idx + 4] = 1
      Hi = Hi @ Fx

      H = np.vstack((H, Hi)) if H is not None else Hi
    # endfor

    Q = config._Q_coeff * np.eye(2 * num_lines)

    K = self.sigma @ H.T @ np.linalg.inv(H @ self.sigma @ H.T + Q)
    
    dz = Z - expected_Z
    dz[1::2] = normalize_angle(dz[1::2])

    self.mu += K @ dz
    self.mu[2] = normalize_angle(self.mu[2])

    self.sigma = (np.eye(K.shape[0]) - K @ H) @ self.sigma
