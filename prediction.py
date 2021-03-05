## Import the liabraries

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle


## Read the dataset

dataset = pd.read_csv('Toddler_Autism_dataset.csv')

## Encode categorical data

Class_ASD = {'Yes': 1, 'No': 0}
dataset['Class_ASD'] = dataset['Class_ASD'].map(Class_ASD)


## Data Undetstanding

## Count the number of rows and columns in dataset

dataset.shape

## Count the empty values in each column

dataset.isna().sum()

sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')

## Get a count of class of number of '1' and '0'

dataset['Class_ASD'].value_counts() 

dataset['Family_mem_with_ASD'].value_counts()

## Visualization of data
 
sns.countplot(dataset['Class_ASD'], label="Count")

sns.countplot(dataset['Family_mem_with_ASD'], label="Count")

sns.countplot(dataset['Sex'], label="Count")


## Heatmap visualization of the correlation

dataset.corr()

plt.figure(figsize=(15,15))

sns.heatmap(dataset.corr(), annot=True, fmt='.0%')

dataset.info()



## Build Logistic Regression the model

x = dataset.iloc[:, 1:11].values

y = dataset['Class_ASD'].values

## Feature scaling using MinMaxScaler

## Scaling to make independent data to be in range 0-1 or 0-100

sc = MinMaxScaler(feature_range = (0,1))

dataset_scaled = sc.fit_transform(x)

dataset_scaled = pd.DataFrame(dataset_scaled)

## Split the data into training dataset and testing dataset

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)


## Data modelling

lr = LogisticRegression(solver = 'lbfgs', max_iter = 200)

lr.fit(x_train,y_train)

## Model accuracy

y_predict = lr.predict(x_test)

print('\n\n The Accuracy is:', lr.score(x_train, y_train))

## Save the model in disk


pickle.dump(lr, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


