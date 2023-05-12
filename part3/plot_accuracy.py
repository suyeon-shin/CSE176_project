import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time 
import seaborn as sns
import matplotlib.pyplot as plt

start_time = time.time()

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Store all features in an array
feature_array = ['headline', 'short_description', 'authors']

# Dictionary to store training and testing accuracies
accuracies = {'Train': [], 'Test': []}

# Loop through all the features in the feature array
for features in feature_array:

    # Preprocess the text data using CountVectorizer or TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[features])
    y = data['category']

    # Split the data into the training and testing sets
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

    # Define the LightGBM model
    cl = lgb.LGBMClassifier(**params)

    # Use training data to train the model
    cl.fit(X_train, y_train)

    # Test the model on the training set
    y_train_pred = cl.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    accuracies['Train'].append(train_accuracy)

    # Test the model on the testing set
    y_test_pred = cl.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    accuracies['Test'].append(test_accuracy)

    # Save the classification report
    report = classification_report(y_test, y_test_pred)
    print(features, "Accuracy: {:.2f}%".format(test_accuracy * 100))
    #print(report)

# Plot training and testing accuracies
plt.figure(figsize=(8, 6))
sns.lineplot(x=feature_array, y=accuracies['Train'], marker='o', label='Train Accuracy')
sns.lineplot(x=feature_array, y=accuracies['Test'], marker='o', label='Test Accuracy')
plt.title('Training and Testing Accuracies')
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
