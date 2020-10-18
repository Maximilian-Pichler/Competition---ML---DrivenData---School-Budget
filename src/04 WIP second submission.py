# %%
import os
import time
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from multilabel import multilabel_train_test_split

import concurrent.futures


# %% path configuration
path_raw = Path('../data/raw')
path_interim = Path('../data/interim')
path_processed = Path('../data/processed')

# %% Data formatting and transforming
df = pd.read_csv(f'{path_raw}/TrainingData.csv')

# splitting numeric, string and label columns
##numerical columns
num = ['FTE', 'Total']
##text columns
text = ['Facility_or_Department','Function_Description','Fund_Description','Job_Title_Description',
'Location_Description','Object_Description','Position_Extra','Program_Description','SubFund_Description',
'Sub_Object_Description','Text_1','Text_2','Text_3','Text_4']
##goal labels (y)
labels = ['Function','Object_Type','Operating_Status','Position_Type',
'Pre_K','Reporting','Sharing','Student_Type','Use']

X = df
X_num = df[num]
X_text = df[text]
y = df[labels]

#transform to category
y = y.astype('category')

# create dummy-values
y = pd.get_dummies(data = y, columns = y.columns, prefix_sep = '__')

#write data to .csv
X_num.to_csv(path_interim / 'X_num.csv', index = False)
X_text.to_csv(path_interim / 'X_text.csv', index = False)
y.to_csv( path_interim / 'y.csv', index = False)
# %% data clreaning
X_num = X_num.fillna(-1000)
X_text = X_text.fillna('')

# %% Trainset splits
X_num_train, X_num_test, _, _ = multilabel_train_test_split(X_num, y, size = 0.2, seed = 123)
X_text_train, X_text_test, y_train, y_test = multilabel_train_test_split(X_text, y, size = 0.2, seed = 123)

# %%
def combine_text_columns(X_train):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_select = set(to_select) & set(X.columns.tolist())
    text_data = X_train[to_select]
    
    # Replace nans with blanks
    text_data.fillna("", inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


# %% [markdown]

# pipeline
#Numeric Pipeline - works
numeric_pipeline = Pipeline([
    ('selector', get_numeric_data),
    ('imputer', SimpleImputer()),
])

#Text Pipeline - WIP
TOKENS_BASIC = '\\S+(?=\\s+)'
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
vec_hash_alphanumeric = HashingVectorizer(norm=None, token_pattern=TOKENS_BASIC,  ngram_range=(1,4), stop_words='english')

text_pipeline = Pipeline([
    ('selector', get_text_data),
    #('vectorizer', vec_alphanumeric),
    ('vectorizer', vec_hash_alphanumeric),
]) 

##multiple model feature-union pipeline
def runModel(model):
    pl = Pipeline([
        ('union', FeatureUnion([
            ('numeric', numeric_pipeline),
            ('text', text_pipeline)
        ])),
        ('clf', OneVsRestClassifier(model()))
    ])
    return pl
# %% run model
#run single-Process fitting
#CLASSIFIERS = [OneVsRestClassifier, KNeighborsClassifier]

def run(Model):
    print(f'load pipeline with {model}-Model')
    pl=runModel(model)
    print(f'start fitting of {model}-Model')
    pl.fit(X_text,y)
    print(f'start predicting with {model}-Model')
    holdout = pd.read_csv(f'{path_raw}TestData.csv', index_col = 0)         #load test data
    predictions = pl.predict_proba(holdout)                                 #predict
    print(f'write {model}-Model predictions')
    predictions_df = pd.DataFrame(columns = y.columns, data = predictions, index = holdout.index)   #output
    predictions_df.to_csv(f'{path_processed}predictions_{model}.csv')                            #output
    print(f'finished prediction with {model}-Model')
# %% run
start = time.time()

run(LogisticRegression)
# %%
