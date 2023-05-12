import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

start_time = time.time()

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Select two features for visualization
feature_array = ['headline', 'short_description']

# Subset the data with the selected features
data_subset = data[feature_array]

# Preprocess the text data using TfidfVectorizer for the selected features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data_subset.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1))
y = data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the parameters for the model
params = {
    'num_leaves': 40,
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'objective': 'multiclass',
    'n_jobs': 4
}

cl = lgb.LGBMClassifier(**params)

cl.fit(X_train, y_train)

# Create a meshgrid for visualization
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Obtain predictions for the meshgrid points
Z = cl.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.8)

# Plot training data points
for class_label in np.unique(y_train):
    plt.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1], label=class_label)

plt.xlabel(feature_array[0])
plt.ylabel(feature_array[1])
plt.title('Decision Boundaries')
plt.legend()
plt.show()

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
