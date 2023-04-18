import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from functions import *

# load data
train_fea, train_gnd, test_fea, test_gnd = load_data()

# extract desired digits (3 and 5)
train_fea_3, train_fea_5, test_fea_3, test_fea_5 = extract_digits(train_fea, train_gnd, test_fea, test_gnd, 3, 5)

# create train, validation, and test set
x_train, y_train, x_validation, y_validation, x_test, y_test = train_validate_test_data(train_fea_3, train_fea_5, test_fea_3, test_fea_5)



# liblinear_l2 = LogisticRegression(solver = "liblinear", penalty = "l2").fit(x_train, y_train)

# predictions = liblinear_l2.predict(x_test)

# # Use score method to get accuracy of model
# score = liblinear_l2.score(x_test, y_test)
# print(score)

# #view confusion matrix
# cm = metrics.confusion_matrix(y_true=y_test, y_pred = predictions, labels = liblinear_l2.classes_)
# print(cm)

# parameter grid
parameters = {
    'penalty' : ['l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['liblinear'],
}

logreg = LogisticRegression()
clf = GridSearchCV(logreg,                    # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=10)                     # number of folds

clf.fit(x_train,y_train)

print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)


logreg = LogisticRegression(C = 1.0, 
                            penalty = 'l2', 
                            solver = 'liblinear')
logreg.fit(x_validation,y_validation)
y_pred = logreg.predict(x_test)
print("Accuracy:",logreg.score(x_test, y_test))



### for visualizing purposes 
### delete after testing
image_size = int(np.sqrt(train_fea.shape[1]))
# display_images(x_train_3.values[:10], f"first 10 3s", image_size)

