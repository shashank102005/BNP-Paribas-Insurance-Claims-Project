# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:11:43 2016

@author: Shashank
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime 
from sklearn.base import TransformerMixin
from scipy import sparse
from scipy.sparse import hstack
from sklearn import svm
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from patsy import dmatrices
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import *
from sklearn.linear_model import LogisticRegression


#################### read the imputed train data
df_train = pd.read_csv('/media/shashank/Data/Projects/Kaggle/BNP Paribas/fixed value imputed data/df_fixed_value_imputed.csv')


################### read the imputed test data
df_test = pd.read_csv('/media/shashank/Data/Projects/Kaggle/BNP Paribas/fixed value imputed data/test_fixed_value_imputed.csv')





colnames = df_train.columns
for i in range(0,len(colnames)):
     print(df_train[colnames[i]].dtype)
     
df_train = df_train.drop(['Unnamed: 0'],axis=1) 
df_test = df_test.drop(['Unnamed: 0'],axis=1)     

df2 = df_train.drop(['target'],axis = 1)


################# create temporary data stacked frame for one hot encoding 
temp_df = pd.concat([df2,df_test],axis=0)


################ make factor variables categorical
colnames =temp_df.columns
for i in range(0,len(colnames)):
    if( temp_df[colnames[i]].dtype == 'object'):
       temp_df[colnames[i]] = pd.Categorical(temp_df[colnames[i]])

for i in range(0,len(colnames)):
     print(temp_df[colnames[i]].dtype)        
   
################ encode categorical variables into dummy variables
colnames =temp_df.columns   
for i in range(0,len(colnames)):
     if( temp_df[colnames[i]].dtype.name == 'category'):
       dummy_ranks = pd.get_dummies(temp_df[colnames[i]], prefix=colnames[i])
       temp_df = temp_df.drop(colnames[i],axis=1)
       temp_df = pd.concat([temp_df,dummy_ranks],join='outer', axis=1)
       
 

####################### separate the one hot encoded train and test data
onehot_train = temp_df[0:len(df_train)]
onehot_test = temp_df[0:len(df_test)]





######################### create a cross validation for finalization of the model ############
target_df= pd.DataFrame(df_train['target'])
X_train,X_val,Y_train,Y_val = train_test_split(onehot_train,target_df,test_size=0.3)



############################ Random Forest Classifier ###########################################    
Y_train= np.ravel(Y_train)
Y_val= np.ravel(Y_val)


model_rf = RandomForestClassifier(n_estimators=2000,max_features='sqrt',oob_score=True,n_jobs=-1)
model_rf.fit(X_train, Y_train)

##### performance on the cross validation set
prob = model_rf.predict_proba(X_val)

logloss = metrics.log_loss(Y_val,prob,normalize=True)  
   
###### feature importance
feat_imp = model_rf.feature_importances_   
feat_imp = sorted(feat_imp, reverse=True) 

##################################################################################################



  





##################### export the one hot encoded data      
onehot_train.to_csv('train_onehot.csv')       
onehot_test.to_csv('test_onehot.csv')       
              
