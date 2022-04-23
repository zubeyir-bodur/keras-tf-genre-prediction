"""
In this script, it is assumed that TV Show Posters were downloaded to the img/orig folder

Given the image folder, down samples the images into the IMDB size 182x268

Below are the training sizes

30 : 55x80
37 : 67x99
44 : 80x118
51 : 93x137
58 : 106x155

So, first, it will create resized (182x268) images into .../img/100 file.
Then those files can be easily down sampled from command line
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas
import os
import os.path as pth
from subprocess import check_output

IMDB_SIZE = (182, 268)
RATIOS = [30, 37, 44, 51, 58]


def main():
    orig_img_path = "../img/orig/"
    imdb_img_path = "../img/100/"
    images = [f for f in os.listdir(orig_img_path) if pth.isfile(pth.join(orig_img_path, f))]

    # Down sample into the IMDB size
    for img_path in images:
        resize_path = imdb_img_path + img_path
        full_path = orig_img_path + img_path
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        resized_img = cv2.resize(img, IMDB_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(resize_path, resized_img)

    """
    First, install ImageMagick for your OS
    After resizing the images with this script, run the following commands for each value of ratio:
        magick mogrify -path "' + down_sample_path + '/" -resize ' + str(ratio) + '% "' + imdb_img_path + '*.jpg"
    I recommend using terminal for this, using os.command() does not seem to work...
    for ratio in RATIOS:
        down_sample_path = "../img/" + str(ratio)
        if not pth.isdir(down_sample_path):
            os.makedirs(down_sample_path)
        command = 'magick mogrify -path "' + down_sample_path + '/" -resize ' + str(ratio) + '% "' + imdb_img_path + '*.jpg"'
        print(command)
        # os.system(command)
        check_output(command, shell=True).decode()
    """


if __name__ == "__main__":
    main()
