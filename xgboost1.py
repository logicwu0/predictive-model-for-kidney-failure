import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import numpy as np
from xgboost import XGBClassifier

data = pd.read_csv("sph6004_assignment1_data.csv")
data['gender'] = data.gender.map({'M': 1, 'F': 0})
mapping = {0: 0, 1: 1, 2: 1, 3: 1}
data['aki'] = data['aki'].map(mapping)
print(data.shape)
# onehot
encoder = OneHotEncoder()
race_encoded = encoder.fit_transform(data[['race']])
columns = encoder.get_feature_names_out(['race'])
race_encoded_df = pd.DataFrame(race_encoded.toarray(), columns=columns)
data = data.drop('race', axis=1)
data = pd.concat([data, race_encoded_df], axis=1)
na_df = data.isnull().sum() * 100 / len(data)
drop_cols = na_df[na_df > 70].keys()
data = data.drop(columns=drop_cols)
data.dropna(axis=1)
data.fillna(data.mean(), inplace=True)
# 去除用户id
data = data.iloc[:, 1:]
print(data.shape)
# 预测值
y = data.iloc[:, 0].values
# 特征
X = data.iloc[:, 1:].values

X = StandardScaler().fit_transform(X)
xgbclf = XGBClassifier()

param_grid_xgbclf = {
    'objective': ['binary:logistic'],
    'learning_rate': [0.1, 0.3, 0.4, 0.5],
    'max_depth': [6, 8, 12, 10],
    'n_estimators': [200],
    'max_leaves': [20, 25],
    'subsample': [0.6, 0.8, 1],
    'random_state': [0, 42, ],
    'lambda': [1, 4],  #L2, increase to make the model more conservative
    #'tree_method':['gpu_hist'], # use GPU
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
grid_search_xgbclf = GridSearchCV(
    xgbclf,
    param_grid=param_grid_xgbclf,
    scoring=['recall', 'f1', 'accuracy', 'roc_auc'],
    refit='f1',
    cv=5,
    n_jobs=-1,  #use all CPU cores to fit trees
    verbose=10
)

start = time()
grid_search_xgbclf.fit(X_train, y_train)
print(f"took {str(time() - start)[:7]} seconds")

print(f"Best parameters: {grid_search_xgbclf.best_params_}")
print(f"Best F1 score: {grid_search_xgbclf.best_score_}")

y_pred_xgbclf = grid_search_xgbclf.best_estimator_.predict(X_test)

print("XGBoost Classification:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgbclf))
print("Precision:", precision_score(y_test, y_pred_xgbclf))
print("Recall:", recall_score(y_test, y_pred_xgbclf))
print("F1-score:", f1_score(y_test, y_pred_xgbclf))
print("ROC AUC:", roc_auc_score(y_test, y_pred_xgbclf))
fpr_xgbclf, tpr_xgbclf, thresholds_xgbclf = roc_curve(y_test, y_pred_xgbclf)