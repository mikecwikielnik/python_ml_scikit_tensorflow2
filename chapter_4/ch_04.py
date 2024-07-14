# coding: utf-8


import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Dealing with missing data

csv_data = '''A,B,C,D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,, 8.0
10.0, 11.0, 12.0,'''

df = pd.read_csv(StringIO(csv_data))
df

df.isnull().sum()

# access the underlying NumPy array
# via the `values` attribute

df.values

# Eliminating training examples or features with missing values

# remove rows that contain missing values

df.dropna(axis=0)

# remove columns that contain missing values

df.dropna(axis=1)

# dropna supports additional params

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with all values NaN)

df.dropna(how='all')

# drop rows that have fewer than 3 real values

df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')

df.dropna(subset=['C'])

# Imputing missing values

# again: our original way

df.values

# impute missing values via the column mean

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data



df.fillna(df.mean())


# Understanding the scikit-learn estimator API

# Handling categorical data

# Nominal and ordinal features

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']

df

# mapping ordinal features

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
df


inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# encoding class labels


# create a mapping dict
# to convert class labels from strings to integers

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping


# to convert class labels from strings to integers

df['classlabel'] = df['classlabel'].map(class_mapping)
df

# reverse the class label mapping

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df

# label encoding with skleaarn's LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

# reverse mapping

class_le.inverse_transform(y)



# Performing one-hot encoding on nominal features

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X


X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()


X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([ ('onehot', OneHotEncoder(), [0]),
                              ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)


# one-hot encoding via pandas

pd.get_dummies(df[['price', 'color', 'size']])


# multicollinearity guard in get_dummies

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)


# multicollinearity guard for the OneHotEncoder

color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([ ('onehot', color_ohe, [0]),
                                ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)                            

## optional: encoding ordinal features

# if we are unsure about the numerical differences between the categories of ordinal features, or the difference between two ordinal values is not defined, we can also encode them using a threshold encoding with 0/1 values. For example, we can split the feature "size" with values M, L, and XL into two new features "x > M" and "x > L". Let's consider the original DataFrame:


df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']
df

# We can use the apply method of pandas' DataFrames to write custom lambda expressions in order to encode these variables using the value-threshold approach:

df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)

del df['size']
df

# Partitioning a dataset into a separate training and test set

df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color Intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline'] 

print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,  random_state = 0, stratify = y)


# Bringing features onto the same scale

# NORMALIZATION

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)



# STANDARDIZATION

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# A manual example

ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())

print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))



# Selecting meaningful features

# L1 and L2 regularization as penalites against model complexity

# A geometric interpretation on L2 regularization



# Sparse solutions with L1-regularization



# For regularized models in scikit-learn that support L1 regularization, 
# we can simply set the `penalty` param to `'l1'` to obtain a sparse solution:


LogisticRegression(penalty = 'l1', solver = 'liblinear', multi_class=  'ovr')


lr = LogisticRegression(penalty= 'l1', C=1.0, solver = 'liblinear', multi_class = 'ovr')
# C=1.0 is default. You can increase or decrease it 
# to make the regularization effect stronger or weaker, respectively
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy', lr.score(X_test_std, y_test))



lr.intercept_