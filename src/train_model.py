"""
CS 484 - Introduction to Computer Vision - Term Project
Genre Prediction for TV Series

Adopted From: Nirman Dave, https://github.com/nddave/Movie-Genre-Prediction
"""

import model
import data_load
from data_load import load_data

# TODO
#  Decide what parameters should be tuned
#  Can decrease the number of hyper parameters
#  to decrease the training time
epochs = 50
# These can be discarded imo
min_year = 1997
max_year = 2017
# New shrink ratios
ratios = [30, 37, 44, 51, 58]
# Old shrink ratios - his optimal ratio was 40 %
# ratios = [30, 40, 50, 60, 70]
# New learning rate hyper parameters
lrates = [1e-3, 5e-4, 1e-4, 5e-4, 1e-5]
# Old learning rate parameters
# His optimal lrate was 1e-4
# lrates = [0.001, 1e-4, 1e-5, 1e-6, 1e-7]
# New genre: Reality-TV is added
genres = ['Horror', 'Romance', 'Action', 'Documentary', 'Reality-TV']
# genres = ['Horror', 'Romance', 'Action', 'Documentary']


def main(ratios_):
    """
    Build the model across all hyperparameter settings
    """
    for ratio in ratios_:

        x_train, y_train = load_data(min_year, max_year, genres, ratio, 'train')
        x_val, y_val = load_data(min_year, max_year, genres, ratio, 'validate')

        for lr in lrates:
            model.build(1, min_year, max_year, genres, ratio, epochs, lr,
                        x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val,
                        )


if __name__ == '__main__':
    main(ratios)
