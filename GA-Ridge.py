# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 加载数据和配置Matplotlib
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

data = pd.read_excel('回归数据.xlsx')
X = data.iloc[:, :-1]  # 特征列
y = data.iloc[:, -1]  # 标签
from sklearn.preprocessing import MinMaxScaler
# 归一化特征数据
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# 定义超参数范围
alpha_list = np.logspace(-3, 1, 10).tolist()  # 从0.001到10
solver_list = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']  # 求解器
tol_list = np.logspace(-5, -2, 10).tolist()  # 收敛精度从1e-5到1e-2
max_iter_list = np.arange(2000, 10001, 100).tolist()  # max_iter从100到1000

# 遗传算法超参数
POPULATION_SIZE = 50  # 种群数量
MAX_GENERATIONS = 50  # 最大迭代次数
CXPB = 0.5  # 交叉概率
MUTPB = 0.3  # 变异概率

# 重置遗传算法类
if "FitnessMax" in dir(creator):
    del creator.FitnessMax
if "Individual" in dir(creator):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_alpha", np.random.choice, alpha_list)
toolbox.register("attr_solver", np.random.choice, solver_list)
toolbox.register("attr_tol", np.random.choice, tol_list)
toolbox.register("attr_max_iter", np.random.choice, max_iter_list)

# 个体定义
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_alpha, toolbox.attr_solver, toolbox.attr_tol, toolbox.attr_max_iter), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 更新评估函数
def evalRidge(individual):
    alpha = individual[0]
    solver = individual[1]
    tol = individual[2]
    max_iter = individual[3]

    # 数据划分，random_state设置为0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 模型定义，random_state设置为0
    model = Ridge(
        alpha=alpha,
        solver=solver,
        tol=tol,
        max_iter=max_iter,  # 添加 max_iter 参数
        random_state=0
    )

    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    r2_train = r2_score(y_train, predictions_train)
    r2_test = r2_score(y_test, predictions_test)

    return (1 * r2_train + 0 * r2_test,)  # 只用训练集的R2来优化

toolbox.register("evaluate", evalRidge)

# 定义交叉和变异
toolbox.register("mate", tools.cxTwoPoint)

# 针对数值型参数进行高斯变异
def customMutate(individual):
    # 对 alpha 和 tol 进行高斯变异
    if isinstance(individual[0], float):  # alpha
        individual[0] = np.random.uniform(0.001, 10)
    if isinstance(individual[2], float):  # tol
        individual[2] = np.random.uniform(1e-5, 1e-2)
    if isinstance(individual[3], int):  # max_iter
        individual[3] = np.random.randint(100, 1001)
    return individual,

toolbox.register("mutate", customMutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=MAX_GENERATIONS,
                                   stats=stats, halloffame=hof, verbose=True)

    # 可视化R方随遗传算法迭代次数的变化
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
    best_alpha = best_individual[0]
    best_solver = best_individual[1]
    best_tol = best_individual[2]
    best_max_iter = best_individual[3]
    print("最优超参数：alpha={}, solver={}, tol={}, max_iter={}".format(best_alpha, best_solver, best_tol, best_max_iter))

    # 重新计算最优模型的R2分数
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = Ridge(alpha=best_alpha,
                  solver=best_solver,
                  tol=best_tol,
                  max_iter=best_max_iter,
                  random_state=0)
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    weighted_r2 = 0.5 * r2_train + 0.5 * r2_test
    print("训练集 R2: {:.3f}, 测试集 R2: {:.3f}, 加权平均 R2: {:.3f}".format(r2_train, r2_test, weighted_r2))
