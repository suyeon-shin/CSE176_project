from functions import *

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# These are for the graph
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# load data
train_fea, train_gnd, test_fea, test_gnd = load_data()

# extract desired digits (3 and 5)
train_fea_3, train_fea_5, test_fea_3, test_fea_5 = extract_digits(train_fea, train_gnd, test_fea, test_gnd, 3, 5)

# create train, validation, and test set
x_train, y_train, x_validation, y_validation, x_test, y_test = train_validate_test_data(train_fea_3, train_fea_5,
                                                                                        test_fea_3, test_fea_5)

# Parameters for the random forests
params = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 40), 'min_samples_split': randint(2, 10)}

# Create classification forest
forest = RandomForestClassifier()

# Find optimal hyperparameters using randomized search
clf = RandomizedSearchCV(forest, param_distributions=params, n_iter=5, cv=5, return_train_score=True)

clf.fit(x_train, y_train)
best_rf = clf.best_estimator_

# Train the model with the data and randomized search hyperparamters
# clf.fit(x_train, y_train)

# Now with validation set
#clf.fit(x_validation, y_validation)

print("Tuned Hyperparameters :", clf.best_params_)  # Display best hyperparameters
print("Accuracy :", clf.best_score_)  # Display the accuracy

# Display the accuracy on the test set
print("Random Forests Test Accuracy:", metrics.accuracy_score(y_test, clf.best_estimator_.predict(x_test)))

# Display confusion matrix
print(metrics.confusion_matrix(y_test, clf.best_estimator_.predict(x_test)))

# plot the training error and validation error for different hyperparameters
train_errors = 1 - clf.cv_results_['mean_train_score']
val_errors = 1 - clf.cv_results_['mean_test_score']
test_errors = 1 - metrics.accuracy_score(y_test, best_rf.predict(x_test))
n_estimators_list = [str(p['n_estimators']) for p in clf.cv_results_['params']]
plt.plot(n_estimators_list, train_errors, label='Training Error')
plt.plot(n_estimators_list, val_errors, label='Validation Error')
plt.plot(n_estimators_list, [test_errors] * len(clf.cv_results_['params']), label='Test Error', linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('Error')
plt.legend()
plt.show()
