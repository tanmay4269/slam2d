"""
Based off of paper: https://www.joace.org/uploadfile/2014/0113/20140113054354731.pdf
and shares concepts from slam course: http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam05-ekf-slam.pdf
"""

import numpy as np
import utils
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
    """
    Arguments:
    - curr_odom
    - odom_cov
    - dt

    - lines
    - curr_landmark_ids

    1. prediction step:
      - mu: self
      - sigma: self
      - odom: args
      - odom_cov: args
      - dt: args

    2. correction step:
      - mu: self
      - sigma: self
      - lines: slam.landmarks (in bot's local frame)
      - current_landmark_ids: slam.feature_extractor
      - observed_landmarks: self
      - [FILL IN!]

    Return:
    - [FILL IN!]
    """

    if self.observed_landmarks is None:
      self.observed_landmarks = np.zeros(
        np.max(curr_landmark_ids)
      )

    self.prediction_step(curr_odom, odom_cov, dt)
    self.correction_step(local_lines, curr_landmark_ids)

    return (self.mu, self.sigma)
    

  def prediction_step(odom, odom_cov, dt):
    """
    This step exclusively predicts bot's pose, it's mean and covariance
    - mu: pose at t-1
    - odom: odom at t
    """

    theta = self.mu[2]
    v = odom[0]
    w = odom[1]
    
    self.mu[0] += -v/w * np.sin(theta) + v/w * np.sin(theta + w * dt)
    self.mu[1] +=  v/w * np.cos(theta) - v/w * np.cos(theta + w * dt)
    self.mu[2] += w * dt

    self.mu[2] = normalize_angle(self.mu[2])

    G = np.eye(self.mu.shape[0])
    G[0, 2] = -v/w * cos(theta) + v/w * cos(theta + w * dt)
    G[1, 2] = -v/w * sin(theta) + v/w * sin(theta + w * dt)

    self.sigma = G * self.sigma * G.T
    self.sigma[0:3, 0:3] += odom_cov


  def correction_step(lines, current_landmarks_ids):
    """
    mu: x, y, theta of bot and (rho, alpha) for all landmarks in global frame
    lines: in local frame
    """

    # WARNING! - padding may messup
    """ 
    zero padding:
      - observed_landmarks 
      - mu, sigma
    """
    i_max = np.max(current_landmarks_ids)
    pad = max(0, i_max - self.observed_landmarks.shape[0])
    self.observed_landmarks = np.array([self.observed_landmarks, np.zeros(pad)])
    
    N_old = self.mu.shape[0]
    self.mu = np.array([self.mu, np.zeros(pad)])
    self.sigma = np.pad(self.sigma, ((0, pad), (0, pad)), 
      mode="constant", constant_value=0)

    # WARNING!
    N_new = self.mu.shape[0]
    for i in range(N_old, N_new):
      self.sigma[i,i] = np.inf 

    # Z === readings at time = t
    # Z is (rho, alpha) in "LOCAL FRAME"
    # expected_Z is predicted Z from info from t-1 and pediction_step
    num_lines = lines.shape[0]
    Z = np.zeros(2 * num_lines)
    expected_Z = np.zeros(2 * num_lines)

    # stacked Jacobian blocks 
    # will be of shape: (2m, 3 + 2N), N being total number of landmarks
    H = None

    for idx in current_landmarks_ids:
      # for new landmarks
      if self.observed_landmarks[idx] == 0:
        rho, alpha = utils.line2polar(lines[idx])

        self.mu[2 * idx + 3] = rho + \
          self.mu[0] * np.cos(alpha) + \
          self.mu[1] * np.sin(alpha)

        self.mu[2 * idx + 4] = normalize_angle(self.mu[2] + alpha)

        self.observed_landmarks[idx] = 1
      
      rho, alpha = utils.line2polar(lines[idx])  # in local frame

      Z[2 * idx] = rho
      Z[2 * idx + 1] = alpha

      expected_Z[2 * idx] = self.mu[2 * idx] - \
        (self.mu[0] * np.cos(alpha) + self.mu[1] * np.sin(alpha))

      expected_Z[2 * idx + 1] = normalize_angle(self.mu[2 * idx + 1] - self.mu[2])

      Hi = np.array([
        [-cos(alpha), -sin(alpha), 0, 1, (self.mu[0] * sin(alpha) - self.mu[1] * cos(alpha))],
        [0, 0, -1, 0, 1]
      ])

      Fx = np.zeros((5, N_new))
      Fx[:3, :3] = np.eye(3)
      Fx[3, 2 * idx + 3] = 1
      Fx[4, 2 * idx + 4] = 1
      Hi = Hi @ Fx

      if H is not None:
        H = np.vstack((H, Hi))
      else:
        H = Hi
    # endfor

    Q = config._Q_coeff * np.eye(2 * num_lines)

    K = self.sigma @ H.T @ np.linalg.inv(H @ self.sigma @ H.T + Q)
    
    dz = Z - expected_Z
    dz[1::2] = normalize_angle(dz[1::2])

    self.mu += K @ dz
    self.mu[2] = normalize_angle(self.mu[2])

    self.sigma = (np.eye(K.shape[0]) - K @ H) @ self.sigma
