# CSE 176 Project Part 3

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization
import json
from sklearn.model_selection import train_test_split
import lightgbm as lgb

data_set = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Summary of data set
data_set.info()

X = data_set[['link', 'headline', 'short_description', 'authors', 'date']]  # Declare the feature vectors
y = data_set['category']  # Declare the target variable


# X = data_set.drop(['link', 'headline', 'short_description', 'authors'], axis=1)


# X = pd.get_dummies(data_set[['link']])
# X = pd.get_dummies(data_set[['headline']])
# X = pd.get_dummies(data_set[['short_description']])
# X = pd.get_dummies(data_set[['authors']])
# X = pd.get_dummies(data_set[['date']])
# y = pd.get_dummies(data_set[['category']])



# Separate the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create the lightgbm model
clf = lgb.LGBMClassifier()
# clf.fit(X_train, y_train)
