"""
CS 484 - Introduction to Computer Vision - Term Project
Genre Prediction for TV Series

Adopted From: Nirman Dave, https://github.com/nddave/Movie-Genre-Prediction
"""

import model
import pandas as pd
import numpy as np
from PIL import Image
import io
import glob
from sklearn.model_selection import train_test_split
import os


epochs = 50
# These can be discarded imo
min_year = 1997
max_year = 2017
# New shrink ratios
ratios = [44] # Use 44 for now
# ratios = [30, 37, 44, 51, 58]
# Old shrink ratios - his optimal ratio was 40 %
# ratios = [30, 40, 50, 60, 70]
# New learning rate hyper parameters
# lrates = [1e-3, 5e-4, 1e-4, 5e-4, 1e-5]
lrates = [1e-4] # Use 1 * 10^-4 for now
# Old learning rate parameters
# His optimal lrate was 1e-4
# lrates = [0.001, 1e-4, 1e-5, 1e-6, 1e-7]
# New genre: Reality-TV is added
genres = ['Romance', 'Action', 'Horror', 'Documentary', 'Reality-TV']
# genres = ['Horror', 'Romance', 'Action', 'Documentary']


def img_to_rgb(img_file_path):
    """
    Takes an image of 3 channels (RGB) can convert it to a row of pixels.
    """
    data = open(img_file_path, 'rb').read()
    img = Image.open(io.BytesIO(data))
    rgb_img = img.convert('RGB')
    pixels = []
    for i in range(img.size[0]):
        row = []
        for j in range(img.size[1]):
            r, g, b = rgb_img.getpixel((i, j))
            pixel = [r / 255, g / 255, b / 255]
            row.append(pixel)
        pixels.append(row)
    return pixels


def main(ratios_):
    """
    Build the model across all hyperparameter settings
    """
    for ratio in ratios_:

        x = []
        y = []

        dataframe = pd.read_csv('../data/484da_multihot_cleaned.csv')
        # print(dataframe)
        for index, row in dataframe.iterrows():
            y.append([row['Romance'], row['Action'], row['Horror'], row['Documentary'], row['Reality-TV']])
        y = np.array(y)
        # print(y)
        images_dir = "../img/" + str(ratio) + "/"
        globes = glob.glob(images_dir + "*")
        print(f"Found {len(globes)} images")
        for filename in globes:
            x_ = img_to_rgb(filename)
            x.append(x_)
        x = np.array(x)
        print(x.shape)
        print(y.shape)
        if x.shape[0] != y.shape[0]:
          print("X and Y lenghts are not equal.")
          print("This means one image was missing in our dataset")
          print("Now, we will discard the label that was missing, from our Y's")
          for index, row in dataframe.iterrows():
            file_name = "../img/" + str(ratio) + "/" + str(row['ID']) + ".jpg"
            # print(file_name)
            # If the image does not exist
            if not os.path.isfile(file_name):
              print(index)
              # Discard the label - Delete this row from Y's
              y = np.delete(y, index, axis=0)

        print(x.shape)
        print(y.shape)

        # Split the dataset into 3 - Train - Validation - Test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.25, random_state=42)
        test_path = "../cnn_model_results/test"
        x_test_path = test_path + "/x_test.npy"
        y_test_path = test_path + "/y_test.npy"
        if not os.path.isdir(test_path):
          os.makedirs(test_path)

        np.save(x_test_path, x_test)
        np.save(y_test_path, y_test)
        print(x_train.shape)
        print(x_valid.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_valid.shape)
        print(y_test.shape)

        # Start training
        for lr in lrates:
            model.build(1, min_year, max_year, genres, ratio, epochs, lr,
                        x_train=x_train, y_train=y_train,
                        x_val=x_valid, y_val=y_valid)


if __name__ == '__main__':
    main(ratios)
