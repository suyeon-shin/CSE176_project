import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Store all features in an array
feature_array = ['headline', 'short_description', 'authors']

# Loop through all the features in the feature arrau
for features in feature_array:

    # print("=======================================")
    # print("Start")
    # print(features)

    # # If the feature is the data, do this before preprocessing
    # if features == 'date':
    #     data['date'] = data['date'].astype(str)

    # Preprocess the text data using CountVectorizer or TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[features])
    y = data['category']

    # Split the data into the trainging and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the parameters for the model
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05
    }

    # Define the lightgbm model
    cl = lgb.LGBMClassifier(**params)

    # Use training data to train the model
    cl.fit(X_train, y_train)

    # Test the model on the testing set
    y_pred = cl.predict(X_test)

    # Save the report info into a var
    report = classification_report(y_test, y_pred)

    # Access the accuracy score
    accuracy = float(report.split()[-2])

    # Print out the accuracy
    print(features, "Accuracy: {:.2f}%".format(accuracy * 100))

    # print("End")
    # print(features)
    # print("=======================================")