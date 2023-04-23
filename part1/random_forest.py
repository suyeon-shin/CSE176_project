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

# Set fixed hyperparameters
params = {'max_depth': 10, 'min_samples_split': 2}

# Vary n_estimators
n_estimators = [10, 50, 100, 200, 300, 400, 500]

train_errors = []
val_errors = []
test_errors = []

for n in n_estimators:
    # Create classification forest
    forest = RandomForestClassifier(n_estimators=n, **params)

    # Fit model to training data
    forest.fit(x_train, y_train)

    # Calculate training and validation errors
    train_error = 1 - metrics.accuracy_score(y_train, forest.predict(x_train))
    val_error = 1 - metrics.accuracy_score(y_validation, forest.predict(x_validation))
    test_error = 1 - metrics.accuracy_score(y_test, forest.predict(x_test))

    train_errors.append(train_error)
    val_errors.append(val_error)
    test_errors.append(test_error)

# Plot results
plt.semilogx(n_estimators, train_errors, label='Training Error')
plt.semilogx(n_estimators, val_errors, label='Validation Error')
plt.semilogx(n_estimators, test_errors, label='Test Error', linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('Error')
plt.legend()
plt.show()
