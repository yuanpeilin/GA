# 相关模块导入
import random
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Matplotlib配置
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 数据加载
data = pd.read_excel('回归数据.xlsx')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.preprocessing import MinMaxScaler
# 归一化特征数据
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 超参数和遗传算法设置
n_estimators_list = list(range(50, 1001, 50))
learning_rate_list = np.arange(0.01, 0.11, 0.01).tolist()
max_depth_list = list(range(2, 7, 1))

POPULATION_SIZE = 50
MAX_GENERATIONS = 50  # 增加迭代次数
CXPB = 0.5
MUTPB = 0.3

# 删除旧的遗传算法类
if "FitnessMax" in dir(creator):
    del creator.FitnessMax
if "Individual" in dir(creator):
    del creator.Individual

# 注册遗传算法的评估和初始化
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_n_estimators", np.random.choice, n_estimators_list)
toolbox.register("attr_learning_rate", np.random.choice, learning_rate_list)
toolbox.register("attr_max_depth", np.random.choice, max_depth_list)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_estimators, toolbox.attr_learning_rate, toolbox.attr_max_depth), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评估函数
def evalXGB(individual):
    n_estimators = int(individual[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=individual[1], max_depth=int(individual[2]), random_state=0)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    r2_train = r2_score(y_train, predictions_train)
    r2_test = r2_score(y_test, predictions_test)
    return (0.5 * r2_train + 0.5 * r2_test,)

# 注册评估函数到工具箱
toolbox.register("evaluate", evalXGB)

# 交叉、变异和选择操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 交叉和变异后的边界检查
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] < min[i]:
                        child[i] = min[i]
                    elif child[i] > max[i]:
                        child[i] = max[i]
            return offspring
        return wrapper
    return decorator

min_bounds = [min(n_estimators_list), min(learning_rate_list), min(max_depth_list)]
max_bounds = [max(n_estimators_list), max(learning_rate_list), max(max_depth_list)]

toolbox.decorate("mate", checkBounds(min_bounds, max_bounds))
toolbox.decorate("mutate", checkBounds(min_bounds, max_bounds))

# 运行遗传算法
def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    gen = log.select("gen")
    maxs = log.select("max")

    plt.figure()
    plt.plot(gen, maxs, label="R2")
    plt.xlabel("Generations")
    plt.ylabel("R2")
    plt.legend()
    plt.show()

    return pop, log, hof

if __name__ == "__main__":
    _, _, hof = main()
    best_individual = hof[0]
    best_n_estimators = int(best_individual[0])
    best_learning_rate = best_individual[1]
    best_max_depth = int(best_individual[2])
    print("最优超参数：n_estimators={}, learning_rate={}, max_depth={}".format(best_n_estimators, best_learning_rate, best_max_depth))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = XGBRegressor(n_estimators=best_n_estimators, learning_rate=best_learning_rate, max_depth=best_max_depth, random_state=0)
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    weighted_r2 = 0.5 * r2_train + 0.5 * r2_test
    print("训练集 R2: {:.3f}, 测试集 R2: {:.3f}, 加权平均 R2: {:.3f}".format(r2_train, r2_test, weighted_r2))
