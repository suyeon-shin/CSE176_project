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



# with this code i made the file for the order of importances
# with open('feature_importances.txt', 'w') as f:
#     for feature, importance in feature_importances:
#         f.write(f"{feature},{importance}\n")
with open('feature_importances.txt', 'r') as f:
    feature_importances = [line.strip().split(',') for line in f]
selected_features = [feature for feature, _ in feature_importances[:5]]


top_features = 80; 

with open('model_performance.txt', 'w') as f:
    f.write("n_features,test_rmse,execution_time\n")
    for i in range(1, top_features+1):
        selected_features = [feature for feature, _ in feature_importances[:i]]
        model = lgb.LGBMRegressor(**params)
        start = time.time()
        model.fit(X_train[selected_features], y_train)
        end = time.time()
        y_pred = model.predict(X_test[selected_features])
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        execution_time = end - start
        f.write(f"{i},{test_rmse:.2f},{execution_time:.2f}\n")
        print(f"Trained model with {i} features. Test RMSE: {test_rmse:.2f}. Execution time: {execution_time:.2f} seconds")



# print("Number of top features:", top_features )

# selected_features = [feature for feature, _ in feature_importances[:top_features]]

# new_model = lgb.LGBMRegressor(**params)
# new_model.fit(X_train[selected_features], y_train)



# y_pred = new_model.predict(X_test[selected_features])
# accuracy = mean_squared_error(y_test, y_pred, squared=False)
# print(f'Test accuracy: {accuracy:.2f}')


end_time = time.time()

print(f"Execution time: {round(end_time - start_time, 2)} seconds")
