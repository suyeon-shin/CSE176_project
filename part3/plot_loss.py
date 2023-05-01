import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt


start_time = time.time()

# Load the text data into a pandas DataFrame
data = pd.read_json("News_Category_Dataset_v3.json", lines=True)

# Store all features in an array
feature_array = ['link', 'headline', 'short_description', 'authors']

# Preprocess the text data using TfidfVectorizer for all the features
vectorizer = TfidfVectorizer()

#this part is kind of c
X = vectorizer.fit_transform(data[feature_array].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1))
y = data['category']

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

cl = lgb.LGBMClassifier(**params)

cl.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='multi_logloss')

# Get the training and testing losses
train_loss = cl.evals_result_['training']['multi_logloss']
test_loss = cl.evals_result_['valid_1']['multi_logloss']

# Plot the loss curve
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = cl.predict(X_test)

# Save the report info into a variable
report = classification_report(y_test, y_pred)

# Access the accuracy score
accuracy = float(report.split()[-2])

#print(report)

cm = confusion_matrix(y_test, y_pred)
# Save the confusion matrix to a file
np.savetxt("confusion_matrix.txt", cm, fmt='%d')
# print("Confusion matrix:")
# print(cm)
    # pred_accuracy_score = accuracy_score(y_test, y_pred)
    # pred_recall_score = recall_score(y_test, y_pred, average='macro')
    # print('Prediction accuracy', pred_accuracy_score,' recall ', pred_recall_score)

    # cnf_matrix = confusion_matrix(y_test, y_pred, labels=cl.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=cl.classes_)
    # disp.plot()
    # plt.show()

# Print out the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
