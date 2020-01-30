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


from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
for i, val in enumerate(df_test.columns):
    y_train = df[df.columns[-(i+1)]]
    y_test = df_test[df_test.columns[-(i+1)]]
    from sklearn import preprocessing
    from sklearn import utils
    lab_enc = preprocessing.LabelEncoder()

    training_scores_encoded =lab_enc.fit_transform(y_train)
    test_scores_encoded=lab_enc.fit_transform(y_test)
    svm_clf = SVC( random_state=42 , gamma = 'auto', probability= True)
    svm_clf.fit(X_train, training_scores_encoded)
    y_pred_svm = svm_clf.predict(X_test)
#svm_clf = SVC( random_state=42 , gamma = 'auto', probability= True)
    param_grid_svm = [{'kernel': ['rbf' ]  , 'C' : [ 0.5,1]}]
    svm_model=GridSearchCV(svm_clf, param_grid = param_grid_svm, cv=3  , scoring = 'roc_auc')
    #svm_model = GridSearchCV(X = X_train, y= np.ravel(training_scores_encoded),
                           #estimator = svm_clf, param_grid = param_grid_svm, cv=3  , scoring = 'roc_auc')
    from sklearn.metrics import accuracy_score
    svf = balanced_accuracy_score(test_scores_encoded, y_pred_svm)
    svf_acc=accuracy_score(test_scores_encoded, y_pred_svm)
  #  .round(), normalize=Fals
    print("Balanced_accuracy:{:.4f}".format(svf))
    print("accuracy:{:.4f}".format(svf_acc))
   


# Feature importance

from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
%matplotlib inline

plt.tight_layout()
cv = StratifiedKFold(12)
param_range = np.logspace(-6, -1, 12)

viz = ValidationCurve(
    SVC(), param_name="gamma", param_range=param_range,
    logx=True, cv=cv, scoring="f1_weighted", n_jobs=8,
)

viz.fit(X_train, training_scores_encoded)
viz.show()
