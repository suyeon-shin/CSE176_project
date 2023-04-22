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
                   cv=10,                     # number of folds
                   return_train_score=True)   # return train score

clf.fit(x_train,y_train)
best_lr = clf.best_estimator_

print("Tuned Hyperparameters :", clf.best_params_)
print("Accuracy :",clf.best_score_)
# Compare model performance against test set
print("Logistic Regression Test Accuracy:", metrics.accuracy_score(y_test, best_lr.predict(x_test)))

# plot the training error and validation error for different C values
train_errors = 1 - clf.cv_results_['mean_train_score']
val_errors = 1 - clf.cv_results_['mean_test_score']
test_errors = 1 - metrics.accuracy_score(y_test, best_lr.predict(x_test))
plt.semilogx(parameters['C'], train_errors, label='Training Error') # semilogx() Make a plot with log scaling on the x-axis
plt.semilogx(parameters['C'], val_errors, label='Validation Error')
plt.semilogx(parameters['C'], [test_errors] * len(parameters['C']), label='Test Error', linestyle='--')
plt.xlabel('C')
plt.ylabel('Error')
plt.legend()
plt.show()
