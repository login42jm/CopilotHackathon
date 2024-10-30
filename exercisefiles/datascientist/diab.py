# diab.py

# DECISION TREE CLASSIFICATION BASED ON DIABETES DATASET

## INTRODUCTION

# diabetes.csv is originally from the National Institute of Diabetes and Digestive and Kidney
# Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes,
# based on certain diagnostic measurements included in the dataset. Several constraints were placed
# on the selection of these instances from a larger database. In particular, all patients here are females
# at least 21 years old of Pima Indian heritage.2
# From diabetes.csv you can find several variables, some of them are independent
# (several medical predictor variables) and only one target dependent variable (Outcome).

## EXERCISE

# The objective of this project is to build a decision tree classifier based on Scikit-learn and
# Python. The classifier should be able to predict whether a patient has diabetes or not based on
# certain diagnostic measurements included in the dataset.

# 1. Importing Required libraries to build a decision tree classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# 2. Loading the dataset
df = pd.read_csv('/workspaces/CopilotHackathon/exercisefiles/datascientist/diabetes.csv')

# 3. Exploratory Data Analysis
# 3.1. Display the first 5 rows of the dataframe
print(df.head())

# 3.2. Display the number of rows and columns in the dataframe
print(df.shape)

# 3.3. Display the data types of each column
print(df.dtypes)

# 3.4. Display the number of missing values in each column
print(df.isnull().sum())

# 3.5. Display the number of unique values in each column
print(df.nunique())

# 4. Feature Selection
# 4.1. Split the data into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4.2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Building the Decision Tree Classifier
# 5.1. Instantiate the DecisionTreeClassifier class
clf = DecisionTreeClassifier()

# 5.2. Fit the model to the training data
clf.fit(X_train, y_train)

# 5.3. Predict the labels of the test data
y_pred = clf.predict(X_test)

#evalute the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# 5.4. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 6. Visualizing the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
plt.show()

# 7. Conclusion
# The decision tree classifier has been built and evaluated. The accuracy of the model is printed above.
# The decision tree is visualized using matplotlib.