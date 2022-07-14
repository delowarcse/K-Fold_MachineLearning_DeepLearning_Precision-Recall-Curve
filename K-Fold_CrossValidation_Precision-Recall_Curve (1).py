#!/usr/bin/env python
# coding: utf-8

# # This program is the implementation of the K-Fold Cross Validation of Machine Learning and Deep Learning Technique

# In[ ]:


# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets # white = control; red = stroke; wine = data
No_Concussion = pd.read_csv('Healthy Participants Data.csv', delim_whitespace=False)
Yes_Concussion = pd.read_csv('Injured Participants Data.csv', delim_whitespace=False)

# store wine type as an attribute
No_Concussion['data_type'] = 'NoConcussion'   
Yes_Concussion['data_type'] = 'Concussion'

# merge control and stroke data
datas = pd.concat([No_Concussion, Yes_Concussion])
#datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare Training and Testing Datasets
stp_features = datas.iloc[:,:-1]
stp_feature_names = stp_features.columns
stp_class_labels = np.array(datas['data_type'])

X_data = datas.iloc[:,:-1]
y_label = datas.iloc[:,-1]

# Data Normalization
ss = StandardScaler().fit(X_data)
X = ss.transform(X_data)
le = LabelEncoder()
le.fit(y_label)
y = le.transform(y_label)


# In[ ]:


# Libraries for CV, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}


# In[ ]:


# Logistic Regression
# plots 10-fold with darkred and red
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
cv_lr = KFold(n_splits=10, shuffle=True)

y_real_lr = []
y_proba_lr = []

plt.figure(figsize=(6, 6))
i = 0
for train_lr, test_lr in cv_lr.split(X, y):
    probas_lr = LogisticRegression().fit(X[train_lr], y[train_lr]).predict_proba(X[test_lr])
    # Compute ROC curve and area the curve
    precision_lr, recall_lr, _ = precision_recall_curve(y[test_lr], probas_lr[:, 1])
        
    # Plotting each individual PR Curve
    plt.plot(recall_lr, precision_lr, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_lr], probas_lr[:, 1])))
        
    y_real_lr.append(y[test_lr])
    y_proba_lr.append(probas_lr[:, 1])

    i += 1
    
y_real_lr = np.concatenate(y_real_lr)
y_proba_lr = np.concatenate(y_proba_lr)
    
precision_lr, recall_lr, _ = precision_recall_curve(y_real_lr, y_proba_lr)

#plt.plot(recall_lr, precision_lr, color='red',
#            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_lr, y_proba_lr)),
#            lw=2, alpha=0.8)
plt.plot(recall_lr, precision_lr,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_lr, y_proba_lr)),
            lw=2, alpha=0.8)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Logistic Regression PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


cv_dt = KFold(n_splits=10, shuffle=True)

y_real_dt = []
y_proba_dt = []

plt.figure(figsize=(6, 6))
i = 0
for train_dt, test_dt in cv_dt.split(X, y):
    probas_dt = DecisionTreeClassifier(max_depth=4).fit(X[train_dt], y[train_dt]).predict_proba(X[test_dt])
    # Compute ROC curve and area the curve
    precision_dt, recall_dt, _ = precision_recall_curve(y[test_dt], probas_dt[:, 1])
        
    # Plotting each individual PR Curve
    #plt.plot(recall_dt, precision_dt, lw=1, alpha=0.3, color='lightgreen',
    #        label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_dt], probas_dt[:, 1])))
    plt.plot(recall_dt, precision_dt, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_dt], probas_dt[:, 1])))
        
    y_real_dt.append(y[test_dt])
    y_proba_dt.append(probas_dt[:, 1])

    i += 1
    
y_real_dt = np.concatenate(y_real_dt)
y_proba_dt = np.concatenate(y_proba_dt)
    
precision_dt, recall_dt, _ = precision_recall_curve(y_real_dt, y_proba_dt)

plt.plot(recall_dt, precision_dt,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_dt, y_proba_dt)),
            lw=2, alpha=.8)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Decision Tree PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Random Forest Model
cv_rf = KFold(n_splits=10, shuffle=True)

y_real_rf = []
y_proba_rf = []

plt.figure(figsize=(6, 6))
i = 0
for train_rf, test_rf in cv_rf.split(X, y):
    probas_rf = RandomForestClassifier().fit(X[train_rf], y[train_rf]).predict_proba(X[test_rf])
    # Compute ROC curve and area the curve
    precision_rf, recall_rf, _ = precision_recall_curve(y[test_rf], probas_rf[:, 1])
        
    # Plotting each individual PR Curve
    #plt.plot(recall_rf, precision_rf, lw=1, alpha=0.3, color='goldenrod',
    #        label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_rf], probas_rf[:, 1])))
    plt.plot(recall_rf, precision_rf, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_rf], probas_rf[:, 1])))
        
    y_real_rf.append(y[test_rf])
    y_proba_rf.append(probas_rf[:, 1])

    i += 1
    
y_real_rf = np.concatenate(y_real_rf)
y_proba_rf = np.concatenate(y_proba_rf)
    
precision_rf, recall_rf, _ = precision_recall_curve(y_real_rf, y_proba_rf)

#plt.plot(recall_rf, precision_rf, color='gold',
#            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_rf, y_proba_rf)),
#            lw=2, alpha=.8)
plt.plot(recall_rf, precision_rf,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_rf, y_proba_rf)),
            lw=2, alpha=.8)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Random Forest with Hyperparamters Tuning
from sklearn.ensemble import RandomForestClassifier
cv_rft = KFold(n_splits=10, shuffle=True)

y_real_rft = []
y_proba_rft = []

plt.figure(figsize=(6, 6))
i = 0
for train_rft, test_rft in cv_rft.split(X, y):
    probas_rft = RandomForestClassifier(n_estimators=200, max_features='auto').fit(X[train_rft], y[train_rft]).predict_proba(X[test_rft])
    # Compute ROC curve and area the curve
    precision_rft, recall_rft, _ = precision_recall_curve(y[test_rft], probas_rft[:, 1])
        
    # Plotting each individual PR Curve
    #plt.plot(recall_rft, precision_rft, lw=1, alpha=0.3, color='moccasin',
    #        label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_rft], probas_rft[:, 1])))
    plt.plot(recall_rft, precision_rft, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_rft], probas_rft[:, 1])))
                
    y_real_rft.append(y[test_rft])
    y_proba_rft.append(probas_rft[:, 1])

    i += 1
    
y_real_rft = np.concatenate(y_real_rft)
y_proba_rft = np.concatenate(y_proba_rft)
    
precision_rft, recall_rft, _ = precision_recall_curve(y_real_rft, y_proba_rft)

#plt.plot(recall_rft, precision_rft, color='orange',
#            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_rft, y_proba_rft)),
#            lw=2, alpha=.8)
plt.plot(recall_rft, precision_rft,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_rft, y_proba_rft)),
            lw=2, alpha=.8)

#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest with Hyperparameter Tuning PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Support Vector Machine

from sklearn.svm import SVC
cv_svm = KFold(n_splits=10,shuffle=True)
y_real_svm = []
y_proba_svm = []

plt.figure(figsize=(6, 6))
i = 0
for train_svm, test_svm in cv_svm.split(X, y):
    probas_svm = SVC(probability=True).fit(X[train_svm], y[train_svm]).predict_proba(X[test_svm])
    # Compute ROC curve and area the curve
    precision_svm, recall_svm, _ = precision_recall_curve(y[test_svm], probas_svm[:, 1])
        
    # Plotting each individual PR Curve
    #plt.plot(recall_svm, precision_svm, lw=1, alpha=0.3, color='darkcyan',
    #        label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_svm], probas_svm[:, 1])))
    plt.plot(recall_svm, precision_svm, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_svm], probas_svm[:, 1])))
            
    y_real_svm.append(y[test_svm])
    y_proba_svm.append(probas_svm[:, 1])

    i += 1
    
y_real_svm = np.concatenate(y_real_svm)
y_proba_svm = np.concatenate(y_proba_svm)
    
precision_svm, recall_svm, _ = precision_recall_curve(y_real_svm, y_proba_svm)

#plt.plot(recall_svm, precision_svm, color='cyan',
#            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_svm, y_proba_svm)),
#            lw=2, alpha=.8)
plt.plot(recall_svm, precision_svm,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_svm, y_proba_svm)),
            lw=2, alpha=.8)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Support Vector Machine PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# K-Nearest Neighbors Model
from sklearn.neighbors import KNeighborsClassifier
cv_KNN = KFold(n_splits=10,shuffle=True)
y_real_KNN = []
y_proba_KNN = []

plt.figure(figsize=(6, 6))
i = 0
for train_KNN, test_KNN in cv_KNN.split(X, y):
    probas_KNN = KNeighborsClassifier(n_neighbors=4).fit(X[train_KNN], y[train_KNN]).predict_proba(X[test_KNN])
    # Compute ROC curve and area the curve
    precision_KNN, recall_KNN, _ = precision_recall_curve(y[test_KNN], probas_KNN[:, 1])
        
    # Plotting each individual PR Curve
    #plt.plot(recall_KNN, precision_KNN, lw=1, alpha=0.3, color='magenta',
    #        label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_KNN], probas_KNN[:, 1])))
    plt.plot(recall_KNN, precision_KNN, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_KNN], probas_KNN[:, 1])))
    
    y_real_KNN.append(y[test_KNN])
    y_proba_KNN.append(probas_KNN[:, 1])

    i += 1
    
y_real_KNN = np.concatenate(y_real_KNN)
y_proba_KNN = np.concatenate(y_proba_KNN)
    
precision_KNN, recall_KNN, _ = precision_recall_curve(y_real_KNN, y_proba_KNN)

plt.plot(recall_KNN, precision_KNN,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_KNN, y_proba_KNN)),
            lw=2, alpha=.8)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('K-nearest Neighbors PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Deep Neural Network

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import numpy as np

# define k-fold cross validation
KF_dnn = KFold(n_splits=10, random_state=42, shuffle=True)

y_real_dnn = []
y_proba_dnn = []

plt.figure(figsize=(6, 6))
i = 0
for i, (train_dnn, test_dnn) in enumerate(KF_dnn.split(X, y)):
    #create model
    model_dnn = Sequential()
    model_dnn.add(Dense(12, input_dim=79, activation='relu'))
    model_dnn.add(Dense(8, activation='relu'))
    model_dnn.add(Dense(1, activation='sigmoid'))

    #compile & fit
    model_dnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model_dnn.fit(X[train_dnn],y[train_dnn], epochs=100, batch_size=10, verbose=0)
                  
    probas_dnn = model_dnn.predict_proba(X[test_dnn]).ravel()
    # Compute ROC curve and area the curve
    precision_dnn, recall_dnn, _ = precision_recall_curve(y[test_dnn], probas_dnn)
        
    # Plotting each individual PR Curve
    #plt.plot(recall_dnn, precision_dnn, lw=1, alpha=0.3, color='royalblue',
    #        label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_dnn], probas_dnn)))
    plt.plot(recall_dnn, precision_dnn, lw=1, alpha=0.3,
            label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y[test_dnn], probas_dnn)))
        
    y_real_dnn.append(y[test_dnn])
    y_proba_dnn.append(probas_dnn)

    i += 1
    
y_real_dnn = np.concatenate(y_real_dnn)
y_proba_dnn = np.concatenate(y_proba_dnn)
    
precision_dnn, recall_dnn, _ = precision_recall_curve(y_real_dnn, y_proba_dnn)

#plt.plot(recall_dnn, precision_dnn, color='blue',
#            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_dnn, y_proba_dnn)),
#            lw=2, alpha=.8)
plt.plot(recall_dnn, precision_dnn,
            label=r'Mean PR (AUC = %0.2f)' % (average_precision_score(y_real_dnn, y_proba_dnn)),
            lw=2, alpha=.8)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Deep Neural Network PR Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Plots all models in a single figure

plt.figure(figsize=(6, 6))
#fig, ax = plt.subplots(1)
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(recall_lr, precision_lr, color='red', label=r'Mean LR PR (AUC=%0.2f)'%(average_precision_score(y_real_lr, y_proba_lr)))
plt.plot(recall_dt, precision_dt, color='green', label=r'Mean DT PR (AUC=%0.2f)'%(average_precision_score(y_real_dt, y_proba_dt)))
plt.plot(recall_rf, precision_rf, color='gold', label=r'Mean RF PR (AUC=%0.2f)'%(average_precision_score(y_real_rf, y_proba_rf)))
plt.plot(recall_rft, precision_rft, color='orange', label=r'Mean RFT PR (AUC=%0.2f)'%(average_precision_score(y_real_rft, y_proba_rft)))
plt.plot(recall_svm, precision_svm, color='cyan', label=r'Mean SVM PR (AUC=%0.2f)'%(average_precision_score(y_real_svm, y_proba_svm)))
plt.plot(recall_KNN, precision_KNN, color='darkmagenta', label=r'Mean KNN PR (AUC=%0.2f)'%(average_precision_score(y_real_KNN, y_proba_KNN)))
plt.plot(recall_dnn, precision_dnn, color='blue', label=r'Mean DNN PR (AUC=%0.2f)'%(average_precision_score(y_real_dnn, y_proba_dnn)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Mean PR Curve for All Models')
plt.legend(loc="lower right")
plt.show()

