# -*- coding: utf-8 -*-
"""
Created on Thu May 19 00:25:39 2022

@author: Luis Torres
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Obtener la data

url = "bank-full.csv"
data = pd.read_csv(url)

# Tratamiento de la data (Limpiar y normalizar la data)

data.housing.replace(['yes', 'no'], [0, 1], inplace=True)


data.marital.replace(['married', 'single', 'divorced'], [0, 1, 2], inplace=True)
data.marital.value_counts()

data.y.replace(['yes', 'no'], [0, 1], inplace=True)
data.y.value_counts()

data.education.replace(['primary', 'secondary', 'tertiary', 'unknown'], [0, 1, 2, 3], inplace=True)
data.education.value_counts()

data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.default.value_counts()

data.loan.replace(['yes', 'no'], [0, 1], inplace=True)
data.loan.value_counts()

data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace=True)
data.contact.value_counts()

data.poutcome.replace(['unknown', 'success', 'failure', 'other'], [0, 1, 2, 3], inplace=True)
data.poutcome.value_counts()


data.drop(['balance', 'duration', 'campaign', 'pdays', 'previous', 'job', 'day', 'month'], axis=1, inplace=True)

age_mean = data.age.mean()
data.age.replace(np.nan, age_mean, inplace=True)

ranges = [0, 8, 15, 18, 25, 40, 60, 100]
ranges_names = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, ranges, labels=ranges_names)
data.age.value_counts()

data.dropna(axis=0, how='any', inplace=True)

# Partir la data en dos
data_train = data[:22606]
data_test = data[22605:]

x = np.array(data_train.drop(['y'], axis=1))
y = np.array(data_test.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], axis=1))
y_test_out = np.array(data_test.y)






