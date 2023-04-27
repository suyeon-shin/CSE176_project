import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time 
from sklearn.decomposition import TruncatedSVD


start_time = time.time()

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Store all features in an array
feature_array = ['link', 'headline', 'short_description', 'authors']

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data[feature_array].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1))
y = data['category']


svd = TruncatedSVD(n_components=100)

X2 = svd.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)

# Define the parameters for the model
params = {
    'objective': 'multiclassova',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_iterations' : 100,
    # 'device_type': 'gpu'
}

cl = lgb.LGBMClassifier(**params)

cl.fit(X_train, y_train)

y_pred = cl.predict(X_test)

report = classification_report(y_test, y_pred)

accuracy = float(report.split()[-2])


print(report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

print("Accuracy: {:.2f}%".format(accuracy * 100))

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")