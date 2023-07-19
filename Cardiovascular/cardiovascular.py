

#this project classifies if a person has cardiovascular disease or not

#import some libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#store a data into a variable dataframe df
df = pd.read_csv('Heart_Disease_Prediction.csv')
#print the first 7 rows of the data
df.head(7)

df.isna().sum()

df['Heart Disease'].value_counts()

mapping = {'Presence' : 1, 'Absence' :0}
df['Heart Disease'] = df['Heart Disease'].map(mapping)
df['Heart Disease'].dtype

#visualizing count data

sns.countplot(x='Heart Disease', data=df)
plt.xlabel('Disease')
plt.ylabel('Count')
plt.title('Distribution of disease')

#visualise the data
sns.countplot(x='Age', hue='Heart Disease', data=df, palette='colorblind', edgecolor=sns.color_palette('dark', n_colors=1))
plt.xticks(rotation=60)
plt.yticks(fontsize=10)

#Get the correlation of the columns
df.corr()



#visualize the correlations
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.0%')

#split the data into featured data and target data
X=df.iloc[:, :-1].values
Y=df.iloc[:, -1].values

#Split the data again into 75% training data set and 25% testing data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=1)

#Feature Scaling
#Scale the values in the data to be values between 0 and 1 inclusive
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#use RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
forest.fit(X_train, Y_train)

#test the model accuracy on the training data set
model = forest
model.score(X_train, Y_train)

#Test the model accuracy on the test dataset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, model.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

#print the confusion matrix
print(cm)
#print the models accuracy on the test data
print('Model test accuracy ={}'.format((TP+TN)/(TP+TN+FN+FP)))
