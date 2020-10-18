import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

path_raw = '/Users/admin/Documents/GitHub/Boxplots for education/data/raw/'
path_interim = '/Users/admin/Documents/GitHub/Boxplots for education/data/interim/'



df = pd.read_csv(f'{path_raw}TrainingData.csv')

colums_sorted = ['FTE',
'Total',
'Facility_or_Department',
'Function_Description',
'Fund_Description',
'Job_Title_Description',
'Location_Description',
'Object_Description',
'Position_Extra',
'Program_Description',
'SubFund_Description',
'Sub_Object_Description',
'Text_1',
'Text_2',
'Text_3',
'Text_4',
'Function',
'Use',
'Sharing',
'Reporting',
'Student_Type',
'Position_Type',
'Object_Type',
'Pre_K',
'Operating_Status']

df = df[colums_sorted]
df.to_csv(f'{path_interim}TrainingData.csv', index = False)


##load data
df = pd.read_csv(f'{path_interim}TrainingData.csv')

#transform to category
X_num = df.iloc[:, :2]
X_alph = df.iloc[:, 2:12]
X_text = df.iloc[:, 12:16]
y = df.iloc[:, 16:]

y = y.astype('category')

#get dummy-values
y = pd.get_dummies(data = y, columns = y.columns, prefix_sep = '_')

#create .csv
X_num.to_csv(f'{path_interim}X_num.csv', index = False)
X_alph.to_csv(f'{path_interim}X_alph.csv', index = False)
X_text.to_csv(f'{path_interim}X_text.csv', index = False)
y.to_csv(f'{path_interim}y.csv', index = False)