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
    print(data)

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
    print(dataset_new)
    # dataset_new.to_csv('../data/484da_cleaned.csv')
    # sketchy = pd.read_csv('../data/484da_cleaned.csv', encoding='ISO-8859-1')
    dataset_genres_only = pd.DataFrame(dictionary_genres_only)
    dataset_for_plots = pd.DataFrame(dictionary_genres_only)
    print(dataset_for_plots)
    # Create a plot with descending order
    sum_df = dataset_for_plots.sum().sort_values(ascending=False)
    print(sum_df)
    sum_df_list = sum_df.tolist()
    genres_indices = [i+1 for i in range(len(genres))]
    plt.figure(figsize=(30, 8))
    plt.bar(x=genres_indices, height=sum_df_list, align='center', width=0.3)
    plot_labels = sum_df.keys().tolist()
    plt.xticks(genres_indices, plot_labels)
    # heatmap = sns.heatmap(dataset_genres_only)
    plt.show()
    return


if __name__ == "__main__":
    main()
