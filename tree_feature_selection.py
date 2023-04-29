import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


start_time = time.time()

# Load the breast cancer dataset
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


model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_importances = [(X.columns[i], importances[i]) for i in range(X.shape[1])]

# Sort the feature importances in descending order
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print the feature importances in descending order
for feature, importance in feature_importances:
    print(f"{feature}: {importance}")

features = [f[0] for f in feature_importances[:10]]
importances = [f[1] for f in feature_importances[:10]]

plt.bar(features, importances)
plt.xticks(rotation=45, ha = 'right')
plt.xlabel('Features')
plt.ylabel('Importance Score(based on feature_importance)')
plt.title('Top 10 Features')
plt.subplots_adjust(bottom=0.3)

custom_labels = ['Entropy Atomic Mass', 'Entropy Thermal Conductivtity', 'Range Atommic Radius', 'std Atomic Radiuys', 'Range FIE',
                 'Electron Affinity', 'STD Thermal Conductivity', 'Entropy Density', 'Range Atomic Mass', 'Range Valence']
plt.xticks(range(10), custom_labels)
plt.show()


y_pred = model.predict(X_test)
accuracy = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test accuracy: {accuracy}')


end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")