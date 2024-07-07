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