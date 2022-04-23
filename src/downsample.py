"""
Given the image folder, down samples the images into the IMDB size 182x268
Then it down samples those IMDB sizes to the training sizes
30 : 55x80
37 : 67x99
44 : 80x118
51 : 93x137
58 : 106x155

So, first, it will create the 182x268 down sampled IMDB images into 100
file. Then it will create those down sampled files
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas

