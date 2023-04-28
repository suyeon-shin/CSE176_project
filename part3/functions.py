import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_data():
    '''
    Load MNISTmini.mat which has (train and test data and labels):
    train_fea, train_gnd, test_fea, test_gnd

    output:
        train_fea, train_gnd, test_fea, test_gnd
        (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series)
    '''
    # load MNISTmini.mat dataset
    data = json.loads("News_Category_Dataset_v3.json") #MNISTmini.mat had: train_fea1, train_gnd1, train_fea1, train_gnd1

    # extract training features and labels
    train_fea = pd.DataFrame(data["train_fea"]) / 255.0
    train_gnd = pd.Series(data["train_gnd"].flatten())   #pd.Series requires a 1D array and flatten() does that
    test_fea = pd.DataFrame(data["test_fea"]) / 255.0
    test_gnd = pd.Series(data["test_gnd"].flatten())

    return train_fea, train_gnd, test_fea, test_gnd