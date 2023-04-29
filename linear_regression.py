import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time 

start_time = time.time()

data_file = pd.read_csv('train.csv')
y = data_file['critical_temp']
X = data_file.drop('critical_temp', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test RMSE: {rmse:.2f}')

end_time = time.time()

print(f"Execution time: {round(end_time - start_time, 2)} seconds")
