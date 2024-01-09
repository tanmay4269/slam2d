import numpy as np
import matplotlib.pyplot as plt

from utils import *
import _config as config
from line_landmark import *

def ransac(raw_data, landmarks_db):

  """
  1. Take the leftmost point and find atleast X points to the right of it
  2. If there doesn't exist these many, discard these points
  3. If they do exist, find best fit line and find nearest points globally
  4. If there aren't many consensus, discard these points
  5. If there are enough consensus, append into LandmarksDB
  """

  data = raw_data.copy()

  deg_per_scan = config._deg_per_scan
  min_line_pts = config._min_line_pts
  ransac_tolerence = config._ransac_tolerence
  ransac_consensus = config._ransac_consensus
  min_line_len = config._min_line_len

  sample = 0
  current_landmarks_ids = []

  while data.shape[1] > ransac_consensus and sample < data.shape[1]:

    sub_samples = []

    angle = data[0, sample]  # assuming `data` is sorted along axis=0 
    central_angle = angle

    sub_sample_idx = sample

    while sub_sample_idx < data.shape[1] and (angle - central_angle) < deg2rad(deg_per_scan):
      
      angle = data[0, sub_sample_idx]

      sub_samples.append(data[:, sub_sample_idx])
      sub_sample_idx += 1

    sub_samples = np.column_stack(sub_samples)
    
    if sub_samples.shape[1] < min_line_pts:
      sample = sub_sample_idx
      continue
    else:
      sampled_column_indices = np.random.choice(sub_samples.shape[1], min_line_pts)
      sub_samples = sub_samples[:, sampled_column_indices]
      data = np.delete(data, sampled_column_indices, axis=1)

    # finding best fit line
    x = sub_samples[1]
    y = sub_samples[2]
    coeff = np.polyfit(x, y, 1)  # slope, y-intercept

    # finding nearest points to line
    x = data[1]
    y = data[2]
    
    distances = np.abs(coeff[0] * x - y + coeff[1]) / np.sqrt(coeff[0]**2 + 1)
    
    within_distance_mask = distances <= ransac_tolerence

    points_within_distance = data[:, within_distance_mask]

    if points_within_distance.shape[1] < ransac_consensus:
      continue
    
    data = data[:, ~within_distance_mask]

    left_idx = np.argmin(points_within_distance[0])
    right_idx = np.argmax(points_within_distance[0])

    line_landmark = np.array([
      [coeff[0], coeff[1]],
      points_within_distance[1:, left_idx].T,
      points_within_distance[1:, right_idx].T
    ])

    line_len = np.linalg.norm(line_landmark[1, :] - line_landmark[2, :])

    if line_len < min_line_len:
      continue

    plt.plot(
      line_landmark[1:, 0], 
      line_landmark[1:, 1], 
      marker='o', 
      color=config._ransac_col, 
      markersize = config._ransac_size
    )

    plt.draw()

    # finally, adding landmark to the database while keeping track of what was currently added 
    current_landmarks_ids.append(
      landmarks_db.append(line_landmark)
    )

  return current_landmarks_ids
