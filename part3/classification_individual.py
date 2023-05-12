import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score
import time 
import seaborn as sns
import matplotlib.pyplot as plt

start_time = time.time()

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Store all features in an array
feature_array = ['headline', 'short_description', 'authors']

# Loop through all the features in the feature arrau
for features in feature_array:

    # Preprocess the text data using CountVectorizer or TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data[features])
    y = data['category']

    # Split the data into the trainging and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the parameters for the model
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',  # why not multi_error ??
        'boosting_type': 'gbdt',    # default: gradient boosting decision tree (gbdt)
        'max_depth' : 6,
        'num_leaves': 36,
        'n_estimators': 100,
        'learning_rate': 0.05,
        'n_jobs' : 4   # this value should be set to the number or real CPU cores (I have 4)
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

    # pred_accuracy_score = accuracy_score(y_test, y_pred)
    # pred_recall_score = recall_score(y_test, y_pred, average='macro')
    # print('Prediction accuracy', pred_accuracy_score,' recall ', pred_recall_score)

    # cnf_matrix = confusion_matrix(y_test, y_pred, labels=cl.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=cl.classes_)
    # disp.plot()
    # plt.show()


end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")