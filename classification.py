# Importing the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Import the iris dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Splitting X and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)


'''GAUSSIAN NAIVE BAYES'''
gnb = GaussianNB()

# Train model
gnb.fit(x_train, y_train)

# Make predictions
gnb_pred = gnb.predict(x_test)

# Print the accuracy
print("Accuracy of Gaussian Naive Bayes: ",
	accuracy_score(y_test, gnb_pred))
# Print other performance metrics
print("Precision of Gaussian Naive Bayes: ",
	precision_score(y_test, gnb_pred, average='weighted'))
print("Recall of Gaussian Naive Bayes: ",
	recall_score(y_test, gnb_pred, average='weighted'))
print("F1-Score of Gaussian Naive Bayes: ",
	f1_score(y_test, gnb_pred, average='weighted'))


'''DECISION TREE CLASSIFIER'''
dt = DecisionTreeClassifier(random_state=0)

# Train the model
dt.fit(x_train, y_train)

# Make predictions
dt_pred = dt.predict(x_test)

# Print the accuracy
print("\nAccuracy of Decision Tree Classifier: ",
	accuracy_score(y_test, dt_pred))
# Print other performance metrics
print("Precision of Decision Tree Classifier: ",
	precision_score(y_test, dt_pred, average='weighted'))
print("Recall of Decision Tree Classifier: ",
	recall_score(y_test, dt_pred, average='weighted'))
print("F1-Score of Decision Tree Classifier: ",
	f1_score(y_test, dt_pred, average='weighted'))


'''SUPPORT VECTOR MACHINE'''
svm_clf = svm.SVC(kernel='linear') # Linear Kernel

# Train the model
svm_clf.fit(x_train, y_train)

# Make predictions
svm_clf_pred = svm_clf.predict(x_test)

# Print the accuracy
print("\nAccuracy of Support Vector Machine: ",
	accuracy_score(y_test, svm_clf_pred))
# Print other performance metrics
print("Precision of Support Vector Machine: ",
	precision_score(y_test, svm_clf_pred, average='weighted'))
print("Recall of Support Vector Machine: ",
	recall_score(y_test, svm_clf_pred, average='weighted'))
print("F1-Score of Support Vector Machine: ",
	f1_score(y_test, svm_clf_pred, average='weighted'))