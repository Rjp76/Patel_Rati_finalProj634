#!/usr/bin/env python
# coding: utf-8

# Rati Patel
# CS634
# Prof Abduallah
# December 4, 2021

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


winedf = pd.read_csv('winequality-red.csv',sep=";")
winedf.head()


# In[3]:


winedf.info()


# In[4]:


#no null values found in dataframe
winedf.isnull().values.all()


# In[5]:


def calc(labs, pred):
    cm = confusion_matrix(labs,pred)
    fp = int((cm.sum(axis=0) - np.diag(cm)).sum() ) 
    fn = int((cm.sum(axis=1) - np.diag(cm)).sum() )
    tp = int(np.diag(cm).sum())
    tn = int(abs(((cm.ravel().sum())*(cm.shape[1])) - (fp + fn + tp)))
    posi = tp + fn
    negi = tn +fp
    tpr = tp/posi
    tnr = tn/negi
    fpr= fp/negi
    fnr = fn / posi
    preci = tp/(tp+fp)
    f1 = (2 *tp)/(2 * tp + fp + fn)    
    acc = (tp+tn)/(posi+negi)
    err = (fp+fn)/(posi + negi)
    bacc = (tpr+tnr)/2
    tss = (tp/(tp+fn))-(fp/(fp+tn))
    hss = (2*((tp*tn)-(fp*fn))/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)))
    indval = [fp,fn,tp,tn,posi,negi,tpr,tnr,fpr,fnr,preci,f1,acc,err,bacc,tss,hss]
    return indval


# In[6]:


def aveCalc(df):
    rst=[]
    for ind,row in df.iterrows():
        val=(row.sum()/len(row))
        rst.append(val)
    return rst


# In[7]:


X= winedf.drop('quality', axis=1)
y = winedf['quality']


# In[8]:


k = 10
kf = KFold(n_splits=k, random_state=None)
fc = 0
ind = ['FP','FN','TP','TN','Positive','Negative','TPR','TNR','FPR','FNR','Precision','F1','Accuracy','Error','BACC','TSS','HSS']

rf_df = pd.DataFrame(index=ind)
svm_df = pd.DataFrame(index=ind)
gru_df = pd.DataFrame(index=ind)


# # Random Forest & SVM

# In[9]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50, random_state=8)


# In[10]:


from sklearn.svm import SVC
model_svm = SVC()


# In[11]:


fc = 0
for train_ind , test_ind in kf.split(X):
    fc=fc+1
    cn= 'fold '+str(fc)
    X_train , X_test = X.iloc[train_ind,:],X.iloc[test_ind,:]
    y_train , y_test = y[train_ind] , y[test_ind]
   
    #Random Forest
    classifier.fit(X_train,y_train)
    RF_pred = classifier.predict(X_test)

    Rf_cal = calc(y_test, RF_pred)
    rf_df[cn]=Rf_cal


  # SVM 
    model_svm.fit(X_train, y_train)
    SVM_pred = model_svm.predict(X_test)
    
    svm_cal = calc(y_test, SVM_pred)
    svm_df[cn]=svm_cal
     

rf_df['Average']=aveCalc(rf_df)
svm_df['Average']=aveCalc(svm_df)


# ## Random Forest Results

# In[12]:


rf_df


# # SVM Results

# In[13]:


svm_df


# # LSTM

# In[14]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import metrics 


# In[15]:


lstm_mod = Sequential()
lstm_mod.add(Dense(12, activation ='softmax', input_shape =(11, )))
lstm_mod.add(Dense(9, activation ='softmax'))
lstm_mod.add(Dense(1, activation ='sigmoid'))
lstm_mod.output_shape
lstm_mod.summary()
lstm_mod.get_config()
  
# List all weight tensors
lstm_mod.get_weights()
lstm_mod.compile(loss ='binary_crossentropy', 
  optimizer ='adamax', metrics = [metrics.categorical_accuracy])


# In[16]:


for train_ind , test_ind in kf.split(X):
    fc=fc+1
    cn= 'fold '+str(fc)
    X_train , X_test = X.iloc[train_ind,:],X.iloc[test_ind,:]
    y_train , y_test = y[train_ind] , y[test_ind]
    lstm_mod.fit(X_train, y_train, epochs = 3,batch_size = 32, verbose = 1,validation_split=0)
    y_pred = lstm_mod.predict(X_test)
    print(y_pred)


# In[ ]:




