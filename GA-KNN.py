import random
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
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
n_neighbors_list = list(range(1, 31))  # K值从1到30
weights_list = ['uniform', 'distance']  # 加权方式
p_list = [1, 2]  # p=1 曼哈顿距离, p=2 欧几里得距离

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
toolbox.register("attr_n_neighbors", np.random.choice, n_neighbors_list)
toolbox.register("attr_weights", np.random.choice, weights_list)
toolbox.register("attr_p", np.random.choice, p_list)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_neighbors, toolbox.attr_weights, toolbox.attr_p), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 定义评估函数
def evalKNN(individual):
    n_neighbors = int(individual[0])
    weights = individual[1]
    p = int(individual[2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    r2_train = r2_score(y_train, predictions_train)
    r2_test = r2_score(y_test, predictions_test)

    # 同时考虑训练集和测试集的R²
    return (0.5 * r2_train + 0.5 * r2_test,)


# 注册评估函数到工具箱
toolbox.register("evaluate", evalKNN)

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


min_bounds = [min(n_neighbors_list), min(weights_list), min(p_list)]
max_bounds = [max(n_neighbors_list), max(weights_list), max(p_list)]

toolbox.decorate("mate", checkBounds(min_bounds, max_bounds))
# 注册变异操作，针对不同的参数类型
def mutate_individual(individual):
    # 对 n_neighbors 和 p 使用 Gaussian 变异
    individual[0] += int(np.round(np.random.normal(0, 1)))  # 对 n_neighbors 进行高斯变异
    individual[2] += int(np.round(np.random.normal(0, 1)))  # 对 p 进行高斯变异

    # 对 weights 使用随机选择变异
    individual[1] = np.random.choice(weights_list)

    # 保证超参数在合法范围内
    individual[0] = max(min(individual[0], max(n_neighbors_list)), min(n_neighbors_list))
    individual[2] = max(min(individual[2], max(p_list)), min(p_list))

    return individual,

# 注册新的变异操作到工具箱
toolbox.register("mutate", mutate_individual)



# 运行遗传算法
def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=MAX_GENERATIONS, stats=stats,
                                   halloffame=hof, verbose=True)

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
    best_n_neighbors = int(best_individual[0])
    best_weights = best_individual[1]
    best_p = int(best_individual[2])
    print("最优超参数：n_neighbors={}, weights={}, p={}".format(best_n_neighbors, best_weights, best_p))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = KNeighborsRegressor(n_neighbors=best_n_neighbors, weights=best_weights, p=best_p)
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    weighted_r2 = 0.5 * r2_train + 0.5 * r2_test
    print("训练集 R2: {:.3f}, 测试集 R2: {:.3f}, 加权平均 R2: {:.3f}".format(r2_train, r2_test, weighted_r2))
