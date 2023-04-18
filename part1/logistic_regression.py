from functions import *

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# load data
train_fea, train_gnd, test_fea, test_gnd = load_data()

# extract desired digits (3 and 5)
train_fea_3, train_fea_5, test_fea_3, test_fea_5 = extract_digits(train_fea, train_gnd, test_fea, test_gnd, 3, 5)

# create train, validation, and test set
x_train, y_train, x_validation, y_validation, x_test, y_test = train_validate_test_data(train_fea_3, train_fea_5, test_fea_3, test_fea_5)


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
best_lr = clf.best_estimator_

print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)

print("Logistic Regression Test Accuracy:", metrics.accuracy_score(y_test, best_lr.predict(x_test)))



#-----------------------------------------------------------------------
# liblinear_l2 = LogisticRegression(solver = "liblinear", penalty = "l2").fit(x_train, y_train)

# predictions = liblinear_l2.predict(x_test)

# # Use score method to get accuracy of model
# score = liblinear_l2.score(x_test, y_test)
# print(score)

# #view confusion matrix
# cm = metrics.confusion_matrix(y_true=y_test, y_pred = predictions, labels = liblinear_l2.classes_)
# print(cm)

### for visualizing purposes 
### delete after testing
image_size = int(np.sqrt(train_fea.shape[1]))
# display_images(x_train_3.values[:10], f"first 10 3s", image_size)

