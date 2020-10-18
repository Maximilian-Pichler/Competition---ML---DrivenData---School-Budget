# %%
import os
import time

import pandas as pd
import numpy as np

from pprint import pprint

from multilabel import multilabel_train_test_split 

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.impute import SimpleImputer


from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

# %%

path_raw = '../data/raw/'
path_interim = '../data/interim/'
path_processed = '../data/processed/'


# %%
#Column Names
##all column names
colums_sorted = ['FTE', 'Total', 'Facility_or_Department','Function_Description','Fund_Description','Job_Title_Description','Location_Description','Object_Description','Position_Extra','Program_Description','SubFund_Description','Sub_Object_Description','Text_1','Text_2','Text_3','Text_4']

##numerical columns
num = ['FTE', 'Total']
##text columns
text = ['Facility_or_Department','Function_Description','Fund_Description','Job_Title_Description','Location_Description','Object_Description','Position_Extra','Program_Description','SubFund_Description','Sub_Object_Description','Text_1','Text_2','Text_3','Text_4']
##goal labels (y)
labels = ['Function','Object_Type','Operating_Status','Position_Type','Pre_K','Reporting','Sharing','Student_Type','Use']
# %%

#read data
df = pd.read_csv(f'{path_raw}TrainingData.csv')

#load data
X = df[colums_sorted]

##y
y = df[labels]
###transform goal-labels (y) to category & get dummy_values
y = y.astype('category')
y = pd.get_dummies(data = y, columns = y.columns, prefix_sep = '__')

#create .csv
df.to_csv(f'{path_interim}TrainingData.csv', index = False)
X.to_csv(f'{path_interim}X.csv', index = False)
X[num].to_csv(f'{path_interim}X_num.csv', index = False)
X[text].to_csv(f'{path_interim}X_text.csv', index = False)
y.to_csv(f'{path_interim}y.csv', index = False)
# %%
start = time.perf_counter()

#read data
X = pd.read_csv(f'{path_interim}X.csv')
X_num = pd.read_csv(f'{path_interim}X_num.csv')
X_text = pd.read_csv(f'{path_interim}X_text.csv')
y = pd.read_csv(f'{path_interim}y.csv')

finish = time.perf_counter
duration = round(finish-start, 2)
print(f'Finished in {duration} seconds')

# %% 
#select Training Columns
num_train = num
text_train = text

#X[num_train].fillna(-1000, inplace = True)
#X[text_train].fillna('', inplace = True)

X_train, X_test, y_train, y_test = multilabel_train_test_split(X[num_train + text_train], y, size = 0.2, seed = 123)
# %%
def combine_text_columns(X, to_select = text_train):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_select = set(to_select) & set(X.columns.tolist())
    text_data = X[to_select]
    
    # Replace nans with blanks
    text_data.fillna("", inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


##functions
get_numeric_data = FunctionTransformer(lambda x: x[num_train], validate = False)
get_text_data = FunctionTransformer(combine_text_columns, validate = False)

# %% pipelines

#Numeric Pipeline - works
numeric_pipeline = Pipeline([
    ('selector', get_numeric_data),
    ('imputer', SimpleImputer()),
])

#Text Pipeline - WIP
TOKENS_BASIC = '\\S+(?=\\s+)'
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
vec_hash_alphanumeric = HashingVectorizer(norm=None, token_pattern=TOKENS_ALPHANUMERIC,  ngram_range=(1,2))

text_pipeline = Pipeline([
    ('selector', get_text_data),
    #('vectorizer', vec_alphanumeric),
    ('vectorizer', vec_hash_alphanumeric),
])
# %% unite numeric and text pipelines with FeatureUnion
pl = Pipeline([
    ('union', FeatureUnion([
        ('numeric', numeric_pipeline),
        ('text', text_pipeline)
    ])),
    ('clf', OneVsRestClassifier(LogisticRegression()))
])
# %% train/fit model
pl.fit(X, y)

# %% predict
holdout = pd.read_csv(f'{path_raw}TestData.csv', index_col = 0)
predictions = pl.predict_proba(holdout)

# %% create results
predictions_df = pd.DataFrame(columns = y.columns, data = predictions, index = holdout.index)
predictions_df.to_csv(f'{path_processed}predictions_whole_set2.csv')
 
# %%
