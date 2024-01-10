import numpy as np
import utils
import _config as config

class EKF:
  def __init__(self):
    self.mu = np.zeros((3, 1))
    self.sigma = np.zeros((3, 3))

  def run(self):
    """
    1. prediction step:
      - 
    2. correction step:
      - provide lines in local frame
    """
    pass

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


  def correction_step(
    mu, sigma
    lines, current_landmarks_ids, observed_landmarks):

    """
    mu: x, y, theta of bot and (rho, alpha) for all landmarks in global frame
    lines: in local frame
    """

    i_max = np.max(current_landmarks_ids)
    pad = i_max - observed_landmarks.shape[0]
    observed_landmarks = np.array([observed_landmarks, np.zeros(pad)])

    # Z contains (rho, alpha) in local frame
    Z = np.zeros(2 * lines.shape[0])
    expected_Z = np.zeros(2 * lines.shape[0])
    H = []

    for idx in current_landmarks_ids:
      # for new landmarks
      if observed_landmarks[idx] == 0:
        rho, alpha = utils.line2polar(lines[idx])
        mu[2 * idx + 3] = rho + mu[0] * np.cos(alpha) + mu[1] * np.sin(alpha)
        mu[2 * idx + 4] = normalize_angle(mu[2] + alpha)

        observed_landmarks[idx] = 1
      
      rho, alpha = utils.line2polar(lines[idx])

      Z[2 * idx] = rho
      Z[2 * idx + 1] = alpha

      expected_Z[2 * idx] = rho - (mu[0] * cos(alpha) + mu[1] * sin(alpha))
      expected_Z[2 * idx + 1] = alpha - mu[2]

      Hi = np.array([
        [-cos(alpha), -sin(alpha), 0, 1, (mu[0] * sin(alpha) - mu[1] * cos(alpha))],
        [0, 0, -1, 0, 1]
      ])

      Fx = np.zeros((5, mu.shape[0]))
      Fx[:3, :3] = np.eye(3)
      Fx[3, 2 * idx + 3] = 1
      Fx[4, 2 * idx + 4] = 1
      Hi = Hi @ Fx

      H = np.array([H, Hi])

    Q = 0.01 * np.eye(2 * lines.shape[0])

    K = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Q)
    
    dz = Z - expected_Z
    dz[1::2] = normalize_angle(dz[1::2])

    mu += K @ dz
    mu[2] = normalize_angle(mu[2])

    sigma = (np.eye(K.shape[0]) - K @ H) @ sigma
