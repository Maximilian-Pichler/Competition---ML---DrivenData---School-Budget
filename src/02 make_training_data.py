import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

##load data
df = pd.read_csv(f'{path_interim}TrainingData.csv')
df = df.drop(columns = ['Text_1','Text_2','Text_3','Text_4'])

#transform to category
X_num = df.iloc[:, :2]
X_label = df.iloc[:, 2:12]
y = df.iloc[:, 12:]

X_label = X_label.astype('category')
y = y.astype('category')

#get dummy-values
X_dummies = pd.get_dummies(data = X_label, columns = X_label.columns, prefix_sep = '_')
y = pd.get_dummies(data = y, columns = y.columns, prefix_sep = '_')

#create .csv
X_num.to_csv(f'{path_interim}X_num.csv', index = False)
X_dummies.to_csv(f'{path_interim}X_dummies.csv', index = False)
y.to_csv(f'{path_interim}y.csv', index = False)