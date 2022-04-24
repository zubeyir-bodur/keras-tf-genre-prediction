"""
CS 484 - Introduction to Computer Vision - Term Project
Genre Prediction for TV Series

Adopted From: Nirman Dave, https://github.com/nddave/Movie-Genre-Prediction
"""
import os
import sys
import data_manage as dm


def download(min_year, ratios, images_dir, original_images_dir):
    """
    Takes in min_year, list of image shrink ratios and image directory
    to sort movie posters by year of origin, download them and save
    to image directory in various different sizes.
    """
    if not os.path.isdir(original_images_dir):
        os.makedirs(original_images_dir)

    dm.download_posters(min_year=min_year)

    for r in ratios:
        path = images_dir + str(r)
        if not os.path.isdir(path):
            os.makedirs(path)
            command = 'mogrify -path "' + path + '/" -resize ' + str(r) + '% "' + original_images_dir + '*.jpg"'
            print(command)
            os.system(command)

    return True


def main():
    """
    Runs the download function over required params.
    """
    min_year = 1997
    resizes = [30, 40, 50, 60, 70]
    images_dir = '../../img/'
    original_images_dir = '../../img/100/'
    status = download(min_year, resizes, images_dir, original_images_dir)
    return status


if __name__ == '__main__':
    main()
