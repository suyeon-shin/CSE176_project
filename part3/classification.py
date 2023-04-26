# CSE 176 Project Part 3

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization
import json
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Load the json file into a var
data_set = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Create vectorizor
vec = CountVectorizer()
# print(vec)

# Tokenize and count work occurrences in the headlines
# print(data_set['headline'])
corpus = data_set['headline']

# Fit the word occurences into the vectorizor and assign to a variable
X = vec.fit_transform(corpus)

# Tokenize all words that are at least two characters long
analyze = vec.build_analyzer()
for headline in data_set['headline']:
    analyze(headline)

# Retrieve words in their array indecies
vec.get_feature_names_out()
X.toarray()


# Transformer for normalizing
transformer = TfidfTransformer(smooth_idf=False)

# Fit the headlines to the transformer
megatron = transformer.fit_transform(X)
print(megatron.toarray()) # Print them

# Does everything that was just done but in one go
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# vectorizer.fit_transform(corpus)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(megatron, data_set['category'], test_size=0.2)

# Create a LightGBM dataset object
train_data = lgb.Dataset(X_train, label=y_train)

# Set hyperparameters for the LightGBM model
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05
}

# Train the LightGBM model
model = lgb.train(params, train_data)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)