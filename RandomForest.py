import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np

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

mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
mask_boolean = np.array(mask, dtype=bool)
print(data.shape)
# 预测值
y = data.iloc[:, 0].values
# 特征
X = data.iloc[:, 1:].values

X = StandardScaler().fit_transform(X)
X = X[:, mask_boolean]
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 实例化RandomForestClassifier
rfc = RandomForestClassifier()

param_grid_sgd = {
    'n_estimators': [200],
    'max_depth': [None, 2, 10, 8],
    'min_samples_split': [5, 8, 2],
    'min_samples_leaf': [2, 5, 8],
    'max_features': ['sqrt'],
    'criterion': ['entropy'],
    'random_state': [42]
}

# 创建SGDClassifier实例
rf_clf = RandomForestClassifier()

# 定义网格搜索
grid_search_sgd = GridSearchCV(
    rf_clf,
    param_grid=param_grid_sgd,
    # scoring=['recall', 'f1', 'accuracy'],
    refit='f1',
    n_jobs=-1,
    verbose=10
)

start = time()
# 进行网格搜索
grid_search_sgd.fit(X_train, y_train)

end = time()
print(f"took {str(end - start)[:7]} seconds")

# 输出最优参数和最优得分
print(f"Best parameters: {grid_search_sgd.best_params_}")
print(f"Best Recall scores: {grid_search_sgd.best_score_}")
y_pred_lr = grid_search_sgd.best_estimator_.predict(X_test)

# # 输出评价指标
print("RandomForest:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1-score:", f1_score(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
#fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_lr, average=None)
