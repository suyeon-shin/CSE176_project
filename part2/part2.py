# from sklearn import datasets
# import pdb

from functions import *

import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier
# Show all messages, including ones pertaining to debugging
xgb.set_config(verbosity=2)

# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
x_train, y_train, x_test, y_test = load_data()

# Need to transform the y_train value to fit xgboost.
# y_train starts from 0 but not 1
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# pdb.set_trace()
y_train = le.fit_transform(y_train)

# digits = datasets.load_digits()
# images=digits.images
# targets=digits.target
# X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2)
model = XGBClassifier()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# from sklearn.metrics import confusion_matrix
# confusion = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix\n')
# print(confusion)

import numpy as np

# Define a list to store the test errors
test_errors = []

# Iterate over different values of the number of trees
num_trees = [10, 50, 100, 150, 200]  # Update with your desired values
for n in num_trees:
    # Create the XGBoost classifier with the specified number of trees
    model = XGBClassifier(n_estimators=n)
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Calculate the test error
    test_error = 1 - accuracy_score(y_test, y_pred)
    test_errors.append(test_error)

# Plot the test errors as a function of the number of trees
plt.plot(num_trees, test_errors, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Test Error')
plt.title('Test Error vs. Number of Trees')
plt.show()
