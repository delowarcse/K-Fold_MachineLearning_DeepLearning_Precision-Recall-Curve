# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

%matplotlib inline

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

# Libraries for CV, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

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
