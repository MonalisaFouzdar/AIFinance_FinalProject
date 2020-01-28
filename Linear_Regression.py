import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from yellowbrick.model_selection import ValidationCurve, FeatureImportances

# Local variables
train_path = 'data/train_set.csv'
test_path = 'data/test_set.csv'
randomState = 42

# read DataFrame
df = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)

# separate labels
X_train = df[df.columns[:-8]]
X_test = df_test[df_test.columns[:-8]]
# Use copper as test


for i, val in enumerate(df_test.columns):
    y_train = df[df.columns[-(i+1)]]
    y_test = df_test[df_test.columns[-(i+1)]]
    C =np.logspace(0, 4, 10)
    penalty = ['l1', 'l2']
    logistic = LogisticRegression()
    hyperparameters = dict(C=C, penalty=penalty)
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
    print('Training model...')
    best_model = clf.fit(X_train, y_train)
    print('Predicting targets...')
    y_pred_Logisticregression=best_model.predict(X_test)
    #rfc = RandomForestClassifier(n_estimators=8, max_depth=18, random_state=randomState)
    #print('Training model...')
    #rfc.fit(X_train, y_train)
    #print('Predicting targets...')
   # pred_y = rfc.predict(X_test)

    logistic_score = balanced_accuracy_score(y_test, y_pred_Logisticregression)
    print(logistic_score)

# Vizualization Curve, estimators
# viz_rf = ValidationCurve(RandomForestClassifier(), param_name='n_estimators',
#                          param_range=np.arange(5, 30, 1), cv=4, scoring='f1_weighted')
# viz_rf.fit(X_train, y_train)
# viz_rf.show()

# Vizualization Curve, max depth
# viz_rf = ValidationCurve(RandomForestClassifier(n_estimators=15), param_name='max_depth',
#                          param_range=np.arange(5, 20, 1), cv=4, scoring='f1_weighted')
# viz_rf.fit(X_train, y_train)
# viz_rf.show()

# Feature importance
#viz_feat = FeatureImportances(rfc, labels=X_train.columns, relative=False)
viz_feat = FeatureImportances(clf, labels=X_train.columns, relative=False)
viz_feat.fit(X_train, y_train)
viz_feat.show()
