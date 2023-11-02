import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score, accuracy_score
from joblib import dump, load

X = load('../../data-preprocess/run08/data.joblib')
X = X.reshape(X.shape[0], -1)
Y = load('../../data-preprocess/run08/labels.joblib')
# Find indices where Y is 0 or 1
filtered_indices = np.where((Y == 0) | (Y == 9))[0]

# Use these indices to filter X and Y
X = X[filtered_indices]
Y = Y[filtered_indices]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
#X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=1/9, random_state=42)

# Creating the SVM classifier
clf = svm.SVC()

# Training the classifier
clf.fit(X_train, Y_train.ravel())

# Making predictions
Y_pred = clf.predict(X_test)

# Calculating accuracy
acc = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {acc:.2f}")


# Calculating precision
precision = precision_score(Y_test, Y_pred)
print(f"Precision: {precision:.2f}")

# Calculating recall
recall = recall_score(Y_test, Y_pred)
print(f"Recall: {recall:.2f}")

# Calculating F1 score
f1 = fbeta_score(Y_test, Y_pred, beta=1)
print(f"F1 Score: {f1:.2f}")

# Calculating F2 score
f2 = fbeta_score(Y_test, Y_pred, beta=2)
print(f"F2 Score: {f2:.2f}")

