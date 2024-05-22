import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('transfusion_data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 3.0,random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(roc_auc_score(y_test, y_pred))
