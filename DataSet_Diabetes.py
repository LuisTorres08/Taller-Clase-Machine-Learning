# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:42:23 2022

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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

# Regresion Logistica

logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entrenamiento del modelo
logreg.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Regresion Logistica Validacion Cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

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
plt.title("Mariz de confusión Regresion Logistica")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Y Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')

# Maquina De Soporte Vectorial

svc = SVC(gamma='auto')

# Entrenamiento el modelo
svc.fit(x_train, y_train)

# Metricas
print('*'*50)
print('Maquina de Soporte Vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Maquina De Soporte Vectorial Validacion Cruzada

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
print('Maquina soporte vectorial validacion cruzada')

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
plt.title("Mariz de confusión Maquina Soporte Vectorial")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Y Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')



# Arbol De Decision

tree = DecisionTreeClassifier()

# Entrenamiento el modelo
tree.fit(x_train, y_train)

# Metricas
print('*'*50)
print('Arbol de Decision')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {tree.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {tree.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {tree.score(x_test_out, y_test_out)}')

# Arbol De Decision Validacion Cruzada

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
plt.title("Mariz de confusión Arbol de Decision")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Y Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')


# Random Forest

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
plt.title("Mariz de confusión Random Forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1}')

print(f'Y Real: {y_test_out}')
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

print(f'Y Real: {y_test_out}')
print(f'Y Predicho: {y_pred}')


