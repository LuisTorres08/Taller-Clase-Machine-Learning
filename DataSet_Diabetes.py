# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:42:23 2022

@author: Luis Torres
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Obtener la data

url = "diabetes.csv"
data = pd.read_csv(url)

# Tratamiento de la data (Limpiar y normalizar la data)

data.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1, inplace=True)


ranges = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ranges_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
data.Age = pd.cut(data.Age, ranges, labels=ranges_names)
data.Age.value_counts()

data.dropna(axis=0, how='any', inplace=True)

# Partir la data en dos

data_train = data[:384]
data_test = data[384:]

x = np.array(data_train.drop(['Outcome'], axis=1))
y = np.array(data_test.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], axis=1))
y_test_out = np.array(data_test.Outcome)

