import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import SGDClassifier
from deap import base, creator, tools, algorithms
from sklearn.metrics import f1_score, make_scorer
import random

data = pd.read_csv("sph6004_assignment1_data.csv")
data['gender'] = data.gender.map({'M': 1, 'F': 0})
mapping = {0: 0, 1: 1, 2: 1, 3: 1}
data['aki'] = data['aki'].map(mapping)
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
# 预测值
y = data.iloc[:, 0].values
# 特征
X = data.iloc[:, 1:].values

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 设置遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X_train[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    indices = [index for index in range(len(individual)) if individual[index] == 1]
    if len(indices) == 0:
        return (0,)  # 如果没有选中任何特征，返回0适应度
    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    f1 = cross_val_score(clf, X_train[:, indices], y_train, cv=5, scoring=make_scorer(f1_score)).mean()
    return (f1,)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.25)
toolbox.register("select", tools.selTournament, tournsize=7)

# 初始化种群
population = toolbox.population(n=80)

# 运行遗传算法
result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.5, ngen=5, verbose=True)

# 获取最佳解决方案
best_ind = tools.selBest(population, 1)[0]
best_indices = [index for index in range(len(best_ind)) if best_ind[index] == 1]
print("Best Individual: ", best_ind)
print("Best F1 Score: ", best_ind.fitness.values[0])
print("Number of Features Selected: ", len(best_indices))
print("Selected Features Indices: ", best_indices)

# 使用选中的特征训练模型，并在X_test上进行测试，以评估模型在未见数据上的性能
clf = SGDClassifier(max_iter=1000, tol=1e-3).fit(X_train[:, best_indices], y_train)
predictions = clf.predict(X_test[:, best_indices])
f1_test = f1_score(y_test, predictions)
print("F1 Score on Test Data: ", f1_test)
