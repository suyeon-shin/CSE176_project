import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_data():
    '''
    Load MNISTmini.mat which has (train and test data and labels):
    train_fea, train_gnd, test_fea, test_gnd

    output:
        train_fea, train_gnd, test_fea, test_gnd
        (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series)
    '''
    # load MNISTmini.mat dataset
    mnist_mini = loadmat("MNISTmini.mat") #MNISTmini.mat had: train_fea1, train_gnd1, train_fea1, train_gnd1

    # extract training features and labels
    train_fea = pd.DataFrame(mnist_mini["train_fea1"])
    train_gnd = pd.Series(mnist_mini["train_gnd1"].flatten())   #pd.Series requires a 1D array and flatten() does that
    test_fea = pd.DataFrame(mnist_mini["test_fea1"])
    test_gnd = pd.Series(mnist_mini["test_gnd1"].flatten())

    return train_fea, train_gnd, test_fea, test_gnd

def extract_digits(train_fea, train_gnd, test_fea, test_gnd, digit1, digit2):
    '''
    From MNISTmini which contains digits 0-9, extract two digits that we wan to classify

    output:
        train_fea_3, train_fea_5, test_fea_3, test_fea_5
    '''
    # extract images of 3's and 5's
    train_fea_3 = train_fea[train_gnd == digit1]
    train_fea_5 = train_fea[train_gnd == digit2]
    test_fea_3 = train_fea[train_gnd == digit1]
    test_fea_5 = train_fea[train_gnd == digit2]

    return train_fea_3, train_fea_5, test_fea_3, test_fea_5

def train_validate_test_data(train_fea_3, train_fea_5, test_fea_3, test_fea_5):
    '''
    From the 
    '''
    # training set (img 1-1000)
    x_train_3 = train_fea_3[0:1000]
    x_train_5 = train_fea_5[0:1000]
    x_train = np.concatenate((x_train_3, x_train_5))    #double parentheses to pass in as tuple

    y_train_3 = np.full((1000,1), 3, dtype=np.uint8)
    y_train_5 = np.full((1000,1), 5, dtype=np.uint8)
    y_train = np.concatenate((y_train_3, y_train_5)).ravel() #ravel() is used to reshape y_train from a column vector to a 1D array


    # validation set (img 1001-2000)
    x_validation_3 = train_fea_3[1000:2000]
    x_validation_5 = train_fea_5[1000:2000]
    x_validation = np.concatenate((x_validation_3, x_validation_5))

    y_validation_3 = np.full((1000,1), 3, dtype=np.uint8)
    y_validation_5 = np.full((1000,1), 5, dtype=np.uint8)
    y_validation = np.concatenate((y_train_3, y_train_5)).ravel() #ravel() is used to reshape y_train from a column vector to a 1D array


    # test set (img 2001-3000)
    x_test_3 = test_fea_3[0:1000]
    x_test_5 = test_fea_5[0:1000]
    x_test = np.concatenate((x_test_3, x_test_5))

    y_test_3 = np.full((1000,1), 3, dtype=np.uint8)
    y_test_5 = np.full((1000,1), 5, dtype=np.uint8)
    y_test = np.concatenate((y_test_3, y_test_5)).ravel() #ravel() is used to reshape y_test from a column vector to a 1D array

    return x_train, y_train, x_validation, y_validation, x_test, y_test

def display_images(images, title, image_size):
    '''
    this function was created for testing purposes. 
    '''
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    fig.suptitle(title, fontsize=16)
    for idx, img in enumerate(images):
        axes[idx].imshow(img.reshape(image_size, image_size), cmap='gray')
        axes[idx].axis('off')
    plt.show()
