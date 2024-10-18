import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

# 读取数据集
from xgboost import XGBRegressor

file_path = '虚拟真实训练数据.xlsx'
data = pd.read_excel(file_path)

# 检查是否存在空值并使用每列的平均值填充空值
if data.isnull().values.any():
    data = data.fillna(data.mean())

# 获取因子和变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 归一化因子
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 定义基学习器和超参数网格
models = {
    "MLP": MLPRegressor(),
    "KNN": KNeighborsRegressor(),
    "ridge": Ridge(),
    "XGB": XGBRegressor()
}

param_grids = {
    # "MLP": {
    #     "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
    #     "activation": ["relu", "tanh"],
    #     "solver": ["adam", "sgd"],
    #     "learning_rate_init": [0.001, 0.01, 0.1]
    # },
    # "KNN": {
    #     "n_neighbors": [3, 5, 10, 15],
    #     "weights": ["uniform", "distance"],
    #     "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    # },
    # "ridge": {
    #     "alpha": [0.1, 1.0, 10.0, 100.0],
    #     "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    # },
    # "LGBM": {
    #     "n_estimators": [50, 100, 300],
    #     "learning_rate": [0.01, 0.1, 0.3],
    #     "num_leaves": [20, 30, 40, 50, 100],
    #     "verbosity": [-1]
    "MLP": {
        "hidden_layer_sizes": [(50,)],
        "activation": ["relu"],
        "solver": ["sgd"],
        "learning_rate_init": [0.01]
    },
    "KNN": {
        "n_neighbors": [ 5],
        "weights": ["distance"],
        "algorithm": ["auto"]
    },
    "ridge": {
        "alpha": [0.1],
        "solver": [ "svd"]
    },
    "XGB": {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [5]
    }
}

# 评估函数定义
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MSE": mse}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model_evaluations = {}

for name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_

    # 输出最佳参数
    print(f"Best parameters for {name}: {grid_search.best_params_}")

    fold_evaluations = {"R2": [], "MAE": [], "RMSE": [], "MSE": []}

    for train_index, test_index in kf.split(X_scaled):
        X_train_kf, X_test_kf = X_scaled[train_index], X_scaled[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]

        best_model.fit(X_train_kf, y_train_kf)
        fold_evaluation = evaluate_model(best_model, X_test_kf, y_test_kf)

        for key in fold_evaluations:
            fold_evaluations[key].append(fold_evaluation[key])

    model_evaluations[name] = {metric: np.mean(values) for metric, values in fold_evaluations.items()}

# 打印模型评估结果
for model_name, evaluations in model_evaluations.items():
    print(f"Model: {model_name}")
    for metric, score in evaluations.items():
        print(f"{metric}: {score}")
    print("------\n")

optimized_models = {}
bagging_models = {}

for name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    optimized_models[name] = best_model

    # 使用Bagging封装最佳模型
    bagged_model = BaggingRegressor(base_estimator=best_model, n_estimators=10, random_state=42)
    bagging_models[name] = bagged_model

# Bagging模型定义
for name, model in bagging_models.items():
    model.fit(X_scaled, y)

def get_stacked_dataset(models, X, y, kf):
    """
    对于每个模型，使用交叉验证的方式预测整个数据集，以便用作下一层模型的训练数据。
    """
    stack_train = np.zeros((X.shape[0], len(models)))
    for model_idx, (name, model) in enumerate(models.items()):
        for train_index, test_index in kf.split(X):
            clone_model = clone(model)
            clone_model.fit(X[train_index], y[train_index])
            y_pred = clone_model.predict(X[test_index])
            stack_train[test_index, model_idx] = y_pred
    return stack_train

# 使用交叉验证获取基学习器的预测作为新的特征集
X_train_stacked = get_stacked_dataset(optimized_models, X_scaled, y, kf)

# 训练元学习器
meta_learner = LinearRegression()
meta_learner.fit(X_train_stacked, y)

# 评估元学习器
predictions = meta_learner.predict(X_train_stacked)
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print(f"Meta Learner Evaluation: R2={r2}, MAE={mae}, RMSE={rmse}, MSE={mse}")


# # 保存模型
joblib.dump((optimized_models, scaler, meta_learner), 'meta_models.pkl')

# 加载模型
loaded_models, loaded_scaler, loaded_meta_learner = joblib.load('meta_models.pkl')
#
# # 读取无标签数据
unlabeled_file_path = '测试集预测结果.xlsx'
unlabeled_data = pd.read_excel(unlabeled_file_path)

# 处理缺失值，使用前500个样本点的平均值进行填充
for i in range(2, unlabeled_data.shape[0]):
    if unlabeled_data.iloc[i, :].isnull().any():
        start_index = max(2, i - 500)
        end_index = i
        unlabeled_data.iloc[i, :] = unlabeled_data.iloc[start_index:end_index, :].mean()
# 保留经纬度
lat_lon = unlabeled_data.iloc[:, :2]

# 取出特征
features_to_predict = unlabeled_data.iloc[:, 2:]

# 归一化特征
features_scaled = loaded_scaler.transform(features_to_predict)

# 使用基学习器预测生成新的特征集
stacked_features = np.zeros((features_scaled.shape[0], len(loaded_models)))
for model_idx, (name, model) in enumerate(loaded_models.items()):
    stacked_features[:, model_idx] = model.predict(features_scaled)

# 使用元学习器进行最终预测
predictions = loaded_meta_learner.predict(stacked_features)

# 合并经纬度和预测结果
output_data = np.column_stack((lat_lon, predictions))

# 保存为txt文件
np.savetxt('1.txt', output_data, fmt='%f', delimiter='\t')