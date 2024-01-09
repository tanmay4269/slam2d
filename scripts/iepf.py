import numpy as np
import matplotlib.pyplot as plt

from utils import *
import _config as config
from line_landmark import *

def iepf(data, landmarks_db):
  current_landmarks_ids = []

  left_pt = data[1:, 0].T
  right_pt = data[1:, -1].T

  coeff = np.polyfit([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], 1)

  distances = np.abs(coeff[0] * data[1] - data[2] + coeff[1]) / np.sqrt(coeff[0]**2 + 1)
  
  mid_idx = np.argmax(distances)

  if (np.max(distances) < config._iepf_dist_theshold) or (np.max([mid_idx, np.abs(mid_idx - distances.shape[0])]) < config._iepf_samples_threshold):

    line_landmark = np.array([
      [coeff[0], coeff[1]],
      left_pt,
      right_pt
    ])

    plt.plot(
      line_landmark[1:, 0], 
      line_landmark[1:, 1], 
      marker='o', 
      color=config._ransac_col, 
      markersize = config._ransac_size
    )
    
    current_landmarks_ids.append(
      landmarks_db.append(line_landmark)
    )
  

  mid_pt = data[1:, mid_idx].T

  coeff_0 = np.polyfit([left_pt[0], mid_pt[0]], [left_pt[1], mid_pt[1]], 1)
  coeff_1 = np.polyfit([mid_pt[0], right_pt[0]], [mid_pt[1], right_pt[1]], 1)


  line_landmark_0 = np.array([
    [coeff_0[0], coeff_0[1]],
    left_pt,
    mid_pt
  ])

  line_landmark_1 = np.array([
    [coeff_1[0], coeff_1[1]],
    mid_pt,
    right_pt
  ])


  plt.plot(
    line_landmark_0[1:, 0], 
    line_landmark_0[1:, 1], 
    marker='o', 
    color=config._ransac_col, 
    markersize = config._ransac_size
  )

  plt.plot(
    line_landmark_1[1:, 0], 
    line_landmark_1[1:, 1], 
    marker='o', 
    color=config._ransac_col, 
    markersize = config._ransac_size
  )

  current_landmarks_ids.append(
    landmarks_db.append(line_landmark_0)
  )

  current_landmarks_ids.append(
    landmarks_db.append(line_landmark_1)
  )

  return current_landmarks_ids
