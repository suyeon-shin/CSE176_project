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

# Create classification forest
forest = RandomForestClassifier()
forest.fit(x_train, y_train)

# Parameters for the random forests
params = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 40), 'min_samples_split': randint(2, 10)}

# Find optimal hyperparameters using randomized search
clf = RandomizedSearchCV(forest, param_distributions=params, n_iter=5, cv=5)

# Train the model with the data and randomized search hyperparamters
# clf.fit(x_train, y_train)

# Now with validation set
clf.fit(x_validation, y_validation)

print("Tuned Hyperparameters :", clf.best_params_)  # Display best hyperparameters
print("Accuracy :", clf.best_score_)  # Display the accuracy

# Display the accuracy on the test set
print("Random Forests Test Accuracy:", metrics.accuracy_score(y_test, clf.best_estimator_.predict(x_test)))

# Display confusion matrix
print(metrics.confusion_matrix(y_test, clf.best_estimator_.predict(x_test)))

training = []       # Hold vals to be graphed for the training set
validation = []     # Hold vals to be graphed for the validation set
testing = []        # Hold vals to be graphed for the testing set (instructions don't specify needing to plot testing error, so is this needed?)

y_pred = clf.best_estimator_

training.append(mean_squared_error(y_train, forest.predict(x_train)))
validation.append(mean_squared_error(y_validation, clf.predict(x_validation)))
# testing.append(mean_squared_error((y_test, clf.predict(x_test)))) # Again, is this needed?

train_line, = plt.plot(training, color="r", label="Training Score") # Currently can't figure out how to get the 'n_estimators' or 'max_depth' from the hyperparameter tuning to make the lines show up
validation_line, = plt.plot(validation, color='g', label="Validation Score")

plt.legend(handler_map={train_line: HandlerLine2D(numpoints=2)})
plt.ylabel('Mean Squared Error')
plt.xlabel('x val of choice')
plt.show()
