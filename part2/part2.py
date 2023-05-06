# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.preprocessing import LabelEncoder
from functions import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import xgboost as xgb
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt

# Start timer
start_time = time.time()

# Show all messages, including ones pertaining to debugging
xgb.set_config(verbosity=2)

# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
x_train, y_train, x_test, y_test = load_data()

# Need to transform the y_train value to fit xgboost.
# y_train starts from 0 but not 1
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Define the specific hyperparameters we're searching for
param_dist = {'learning_rate': uniform(0, 1),
              'max_depth': randint(1, 10),
              'n_estimators': randint(50, 150),
              }

# Declare the classifier
model = XGBClassifier()

# Declare the random search with our model
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
# Fit the training data to the model
model.fit(x_train, y_train)

# Print out the best hyperparameters
print('Best hyperparameters:', random_search.get_params())

# Test the model
y_pred=model.predict(x_test)

# End the timer and print out the time
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# Get and print out the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Display the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

