# import matplotlib.pyplot as plt

# # Read the data from the file
# filename = 'model_performance.txt'
# data = []
# with open(filename, 'r') as file:
#     file.readline()  # Skip the header line
#     for line in file:
#         data.append(list(map(float, line.strip().split(','))))

# # Extract the relevant columns
# n_features = [row[0] for row in data]
# test_rmse = [row[1] for row in data]
# execution_time = [row[2] for row in data]

# # Plot the data
# # plt.plot(execution_time, test_rmse, marker='o')
# plt.plot(n_features, test_rmse)
# # plt.xlabel('Execution Time (seconds)')
# plt.xlabel('Number of Features')
# plt.ylabel('Test RSME')
# plt.title('Number of Features vs Test RSME')

# plt.show()


import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import time 
import matplotlib.pyplot as plt


start_time = time.time()

data_file = pd.read_csv('train.csv')

y = data_file['critical_temp']
X = data_file.drop('critical_temp', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert the data to LightGBM format
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set the hyperparameters for the LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 50,
    'learning_rate': 0.05,
    'n_estimators': 1000
}

# Train the LightGBM model
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

# Get the feature importances and normalize them
importances = model.feature_importances_
total_importance = sum(importances)
feature_importances = [(X.columns[i], importances[i]/total_importance) for i in range(X.shape[1])]

# Sort the feature importances in descending order
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print the feature importances in descending order and write them to a text file
with open('feature_importances.txt', 'w') as f:
    for feature, importance in feature_importances:
        f.write(f"{feature},{importance}\n")
        print(f"{feature}: {importance}")

# Plot the normalized feature importances
features = [f[0] for f in feature_importances]
importances = [f[1] for f in feature_importances]

plt.bar(features, importances)
plt.xticks([])  # Remove x-axis labels
plt.title('Normalized Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Custom x-axis labels
custom_labels = [feature for feature, _ in feature_importances]
plt.xticks(range(len(custom_labels)), custom_labels, fontsize=6)
plt.show()

end_time = time.time()
print(f"Execution time: {round(end_time - start_time, 2)} seconds")
