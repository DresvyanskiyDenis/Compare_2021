#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocessing utils for labels

This module contains functions for loading, preprocessing and saving labels in .csv files.
Labels are used for emotion recognition task. Example database: SEWA.

To use this module, you need to install pandas and numpy libraries.

List of functions:

    * load_gold_shifted_labels - load gold shifted labels in .csv file format generated by Dmitrii Fedotov.
    * split_labels_dataframe_according_filenames - split loaded labels into different DataFrames according
      to their filename.
    * main - function for testing implemented functions.
"""
from typing import Dict
import pandas as pd

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


def load_gold_shifted_labels(path: str) -> pd.DataFrame:
    """ Read labels generated by Dmitrii and convert them in pandas DataFrame

    :param path: str
            path to the .csv file for reading
    :return: pd.DataFrame
            DataFrame with columns:
                filename
                timestamp
                label_value (arousal or valence)
    """
    # read csv file, first row is trash
    labels_dataframe = pd.read_csv(path, header=None, skiprows=1)
    # split first column in format to two columns filename and timestamp
    # example SEM1002_0.02 -> filename:SEM1002 timestamp:0.02
    labels_dataframe[['filename', 'timestamp']] = labels_dataframe.iloc[:, 0].str.split("_", expand=True)
    # convert timestamp in float
    labels_dataframe['timestamp'] = labels_dataframe['timestamp'].astype('float32')
    # del first row to clear memory
    del labels_dataframe[0]
    # rename columns for convenient use and rearrange them
    labels_dataframe.columns = ['label_value', 'filename', 'timestamp']
    labels_dataframe = labels_dataframe[['filename', 'timestamp', 'label_value']]

    return labels_dataframe


def split_labels_dataframe_according_filenames(label_dataframe:pd.DataFrame) -> Dict[str,pd.DataFrame]:
    """ Split labels located in DataFrame according their filename on different DataFrames

    :param label_dataframe: pd.DataFrame
            DataFrame with labels with columns:
                                              filename
                                              timestamp
                                              label_value (arousal or valence)
    :return: dict
            dict of split DataFrames
    """
    # create resulting variable
    dataframes_list={}
    # extract unique filenames from "filename" column
    filenames=label_dataframe['filename'].unique()
    #split DataFrame according filenames and append split DataFrames to resulting list
    for filename in filenames:
        filename_dataframe=label_dataframe[label_dataframe['filename']==filename]
        dataframes_list[filename]=filename_dataframe
    return dataframes_list


if __name__=="__main__":
    # you can test functions here
    # specify path to file
    path = r"E:\Databases\SEMAINE\SEM_labels_arousal_100Hz_gold_shifted.csv"
    # data should be DataFrame with columns ['filename', 'timestamp', 'label_value']
    data = load_gold_shifted_labels(path)
    # dataframes should be list of DataFrames [pd.DataFrame,...]
    # each with columns ['filename', 'timestamp', 'label_value']
    dataframes=split_labels_dataframe_according_filenames(data)
    print(dataframes)
