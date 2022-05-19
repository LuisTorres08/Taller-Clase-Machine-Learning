# -*- coding: utf-8 -*-
"""
Created on Thu May 19 00:25:39 2022

@author: Luis Torres
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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


# Regresion logistica

# Seleccionar un modelo

logreg = LogisticRegression(solver='lbfgs', max_iter=7600)

# Entrenar el modelo

logreg.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Regresion Logistica')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test_out, y_test_out)}')


# Regresion Logistica Validacion Cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []


for train, test in kfold.split(x, y): 
    logreg.fit(x[train], y[train]) 
    scores_train_train = logreg.score(x[train], y[train]) 
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresión Logística Validación Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión regresion logistica")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')

# Maquina de soporte vectorial

# Seleccionar un modelo

svc = SVC(gamma='auto')

# Entrenar el modelo

svc.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {svc.score(x_test_out, y_test_out)}')


# Maquina de soporte vectorial validacion cruzada

kfold = KFold(n_splits=10) 

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    svc.fit(x[train], y[train]) 
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test]) 
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

print('*'*50)
print('Maquina De Soporte Vectorial Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión maquina soporte vectorial")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')


# Arbol de decisión

# Seleccionar un modelo

tree = DecisionTreeClassifier()

# Entrenar el modelo

tree.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Arbol de decisión')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {tree.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {tree.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {tree.score(x_test_out, y_test_out)}')


# Arbol de Decision validacion cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    tree.fit(x[train], y[train]) 
    scores_train_train = tree.score(x[train], y[train]) 
    scores_test_train = tree.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = tree.predict(x_test_out)

print('*'*50)
print('Arbol de Decision Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {tree.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión arbol de decision")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')

# K-Nearest neighbors

# Seleccionar un modelo

kneighbors = KNeighborsClassifier()

# Entrenar el modelo

kneighbors.fit(x_train, y_train)

# Metricas

print('*'*50)
print('K-Nearest neighbors')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {kneighbors.score(x_test, y_test)}')

# Accuracy de entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {kneighbors.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {kneighbors.score(x_test_out, y_test_out)}')


# K-Nearest neighbors validacion cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    kneighbors.fit(x[train], y[train]) 
    scores_train_train = kneighbors.score(x[train], y[train]) 
    scores_test_train = kneighbors.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = kneighbors.predict(x_test_out)


print('*'*50)
print('K-Nearest neighbors validacion cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {kneighbors.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión Neighbors")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')


# Random Forest

# Seleccionar un modelo

random_forest = RandomForestClassifier()

# Entrenar el modelo

random_forest.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test, y_test)}')

# Accuracy de Entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {random_forest.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test_out, y_test_out)}')


# Random Forest Validacion Cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y): 
    random_forest.fit(x[train], y[train]) 
    scores_train_train = random_forest.score(x[train], y[train]) 
    scores_test_train = random_forest.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = random_forest.predict(x_test_out)

print('*'*50)
print('Random Forest Validacion Cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {random_forest.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred) 
plt.figure(figsize = (6, 6)) 
sns.heatmap(matriz_confusion)
plt.title("Mariz de confusión random forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')





