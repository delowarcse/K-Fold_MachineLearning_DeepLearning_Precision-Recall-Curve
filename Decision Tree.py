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
plt.show()1_score']),np.std(results_dt['test_f1_score'])))
