# CSE 176 Project Part 3

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization
import json
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.feature_extraction import DictVectorizer

# Load the json file into a var
data_set = pd.read_json("News_Category_Dataset_v3.json", lines=True)

X = data_set[['link', 'headline', 'short_description', 'authors', 'date']]  # Declare the feature vectors
y = data_set['category']  # Declare the target variable

# Separate the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# X_train.info()
for feature in X_train:
    X_train[feature] = pd.Series(X_train[feature], dtype="category")
    X_test[feature] = pd.Series(X_test[feature], dtype="category")

y_train = y_train.astype('category')
y_test = y_test.astype('category')


# X_train.info()
# # X_test.info()
# y_train.info()
# # y_test.info()

# Create the lightgbm model
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
