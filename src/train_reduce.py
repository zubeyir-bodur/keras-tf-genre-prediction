"""
CS 484 - Introduction to Computer Vision - Term Project
Genre Prediction for TV Series

"""
import model
import pandas as pd
import numpy as np
from PIL import Image
import io
import glob
import os
from sklearn.model_selection import train_test_split


images_dir = "../img/" 
RATIOS = [30, 37, 44, 51, 58]


def main():
  """
  Deletes the unnecessary images in the img folder
  In other words, reduces the img folder
    - Deletes the images that do not belong to
      484da_multihot_cleaned.csv
  """
  df = pd.read_csv("../data/484da_multihot_cleaned.csv", encoding='ISO-8859-1')
  ID_s = np.array(df['ID'])
  del_count = 0
  for r in RATIOS:
    for filename in glob.glob(images_dir + str(r) + "/*"):
      basename = os.path.basename(filename)
      show_id = int(basename[0:len(basename)-4])
      if show_id not in ID_s:
        # Nothing will be printed if
        # all images are in the final dataset
        print(f"Deleting {filename}; ...")
        # Uncomment to delete 
        os.remove(filename)
        del_count += 1
  print(str(del_count) + " images were deleted.")
  return


if __name__ == "__main__":
  main()