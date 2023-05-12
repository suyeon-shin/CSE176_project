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
from sklearn.metrics import accuracy_score

# Start timer
start_time = time.time()

# Show all messages, including ones pertaining to debugging
xgb.set_config(verbosity=2)

# Load the data and split it into the training and testing sets
x_train, y_train, x_test, y_test = load_data()

# Need to transform the y_train value to fit xgboost.
# y_train starts from 0 but not 1
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Declare the parameters we want to use in the randomized search
# Set the threshold for what they can be
param_dist = {'learning_rate': uniform(0, 1),
              'max_depth': randint(1, 10),
              'n_estimators': randint(50, 150),
              }

# Declare the XGBoost classifier
model = XGBClassifier()

# Randomized search with our XGBoost classifier and parameters for hyperparameter tuning
random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the training data to the model
model.fit(x_train, y_train)

# Print out the best values found for the hyperparameters
print('Best hyperparameters values:', random_search.get_params())

# Test the model using the testing set
y_pred = model.predict(x_test)

# End the timer and print out the time
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# Get and print out the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Define the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Create a figure
figure, ax = plt.subplots()

# Display final confusion matrix
image = ax.imshow(confusion, cmap=plt.cm.Blues)

# Color bar next to the matrix
cbar = ax.figure.colorbar(image, ax=ax)

# Titles/labels
ax.set(xticks=np.arange(confusion.shape[1]),
       yticks=np.arange(confusion.shape[0]),
       xticklabels=le.classes_, yticklabels=le.classes_,
       title="Confusion Matrix",
       ylabel='True label',
       xlabel='Predicted label')

# Make things look nicer
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Annotations
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(j, i, format(confusion[i, j], 'd'),
                ha="center", va="center",
                color="white" if confusion[i, j] > confusion.max() / 2. else "black")

# Display
figure.tight_layout()
plt.show()
