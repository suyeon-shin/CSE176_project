import lightgbm as lgb
import pandas as pd
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#from functions import *
import json

# # JSON file
# f = open ('News_Category_Dataset_v3.json', 'r')
# # Reading from file
# df = json.loads(f.read())

from scipy.sparse import csr_matrix
df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
df['date'] = df['date'].astype(str)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(dtype=float)
data = vec.fit_transform(df.to_dict(orient='records'))
data_sparse = csr_matrix(data)
x_train, x_test, y_train, y_test = train_test_split(data_sparse, df['category'], test_size=0.3, random_state=0)

# # Load the JSON data from a file into a DataFrame
# df = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# # Convert the Timestamp objects in the 'date' column to string objects
# df['date'] = df['date'].astype(str)

# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer()
# # Convert each row to a dictionary
# data = df.to_dict(orient='records')
# df = vec.fit_transform(data).todense()

# x = df[['headline','authors','link','short_description','date']]
# y = df['category']
# # split the dataset into the training set and test set
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)



# Convert the data to LightGBM format
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

# Set the hyperparameters for the LightGBM model
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
}

# Train the LightGBM model
num_rounds = 100
model = lgb.train(params, train_data, num_rounds, valid_sets=[train_data, test_data], early_stopping_rounds=10)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_class = [round(x) for x in y_pred]
accuracy = sum(y_pred_class == y_test) / len(y_test)
print(f'Test accuracy: {accuracy}')