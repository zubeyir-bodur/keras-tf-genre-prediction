"""
CS 484 - Introduction to Computer Vision - Term Project
Genre Prediction for TV Series

This script generates statistics plots
regarding our dataset. By examining these
plots, we will decide
    - which genres should be estimated
    - should genres be merged
    - ...
"""
import pandas as pd
import seaborn as sns
import json
import numpy as np
import matplotlib.pyplot as plt


def str_to_list(string):
    """
    Converts a python to_str of a list
    to an actual list
    :param string:
    :return:
    """
    json_str = "{\"key\":" + string + "}"
    json_str_list = list(json_str)
    for i in range(len(json_str_list)):
        if json_str_list[i] == '\'':
            json_str_list[i] = "\""
    json_str = ''.join(json_str_list)
    return json.loads(json_str)["key"]


def main():
    data = pd.read_csv('../data/484da.csv', encoding='ISO-8859-1')

    # 26 different genres
    genres = []
    genre_vectors = []
    # for each row, save the genres into an array
    for row in data.values:
        genres_row = str_to_list(row[1])
        for genre in genres_row:
            if genre not in genres:
                genres.append(genre)

    for row in data.values:
        vector = [0 for _ in range(len(genres))]
        genres_row = str_to_list(row[1])
        for genre in genres_row:
            idx = genres.index(genre)
            vector[idx] += 1
        genre_vectors.append(vector)

    transposed = np.transpose(np.array(genre_vectors))
    dictionary_genres_only = {}
    dictionary = {"ID": data["Id"].tolist(), "Poster": data["img-url"].tolist()}
    for i in range(len(genres)):
        inf_for_genre_i = transposed[i]
        dictionary[genres[i]] = inf_for_genre_i
        dictionary_genres_only[genres[i]] = inf_for_genre_i
    dataset_new = pd.DataFrame(dictionary)
    # heatmap = sns.heatmap(data)
    # dataset_new.to_csv('../data/484da_cleaned.csv')
    # sketchy = pd.read_csv('../data/484da_cleaned.csv', encoding='ISO-8859-1')
    dataset_for_plots = pd.DataFrame(dictionary_genres_only)
    # Create a plot with descending order
    sum_df = dataset_for_plots.sum().sort_values(ascending=False)
    sum_df_list = sum_df.tolist()
    genres_indices = [i+1 for i in range(len(genres))]
    plt.figure(figsize=(10, 13), dpi=300)
    plt.bar(x=genres_indices, height=sum_df_list, align='center', width=0.6)
    plot_labels = sum_df.keys().tolist()
    plt.xticks(genres_indices, plot_labels, rotation=90)
    plt.xlabel("Genre Names")
    plt.ylabel("Number of TV Series")
    plt.title("Bar Plot for Number of Series per Genre")
    plt.savefig("../stats/barplot_genres_count.png")
    plt.show()

    # Create a DF with aliases as genre names, and columns as genre names

    # Create the co-occurrence matrix
    num_genres = len(genres)
    num_samples = len(data.values)
    co_occurrence_matrix = [[0 for _ in range(len(genres))] for _ in range(len(genres))]
    # for each tv series
    for j in range(num_samples):
        # start from genre i
        for i in range(num_genres):
            # if this series contains genre i
            if dataset_for_plots.values[j][i]:
                # Look for co-occurrences - ignore the diagonals
                for k in range(num_genres):
                    if i != k and dataset_for_plots.values[j][k]:
                        # I co_occurs with K
                        co_occurrence_matrix[i][k] += 1
    plt.figure(figsize=(10, 10), dpi=300)
    ax = sns.heatmap(co_occurrence_matrix, linewidth=0.5)
    plt.xticks(genres_indices, genres, rotation=90)
    plt.yticks(genres_indices, genres, rotation=0)
    plt.title("Genre Co-Occurrence Heatmap for the Dataset")
    plt.savefig("../stats/heatmap.png")
    plt.show()
    return


if __name__ == "__main__":
    main()
