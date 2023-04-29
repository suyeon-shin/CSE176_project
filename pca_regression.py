import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time 
from sklearn.decomposition import PCA


start_time = time.time()

# Load the breast cancer dataset
data_file = pd.read_csv('train.csv')

print(data_file.size)

y = data_file['critical_temp']

X = data_file.drop('critical_temp', axis=1)


pca = PCA(n_components=40)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)

# Convert the data to LightGBM format
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


#setting up pca 


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

y_pred = model.predict(X_test)
accuracy = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test accuracy: {accuracy}')


end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")