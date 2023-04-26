import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Preprocess the text data using CountVectorizer or TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['headline'])
y = data['category']

# Split the preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define a LightGBM model and its hyperparameters
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05
}

model = lgb.LGBMClassifier(**params)

# Train the LightGBM model on the training data
model.fit(X_train, y_train)

# Evaluate the trained model on the testing data
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))