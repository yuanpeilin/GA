# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
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
activation_list = ['relu', 'tanh', 'logistic']
hidden_layer_sizes_list = [(10,), (50,), (100,), (50, 50), (100, 100)]  # 单层或双层神经元
learning_rate_init_list = np.linspace(0.001, 0.1, 10).tolist()  # 学习率从0.001到0.1
solver_list = ['lbfgs', 'sgd', 'adam']  # 优化器

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
toolbox.register("attr_activation", np.random.choice, activation_list)
toolbox.register("attr_hidden_layer_sizes", np.random.choice, hidden_layer_sizes_list)
toolbox.register("attr_learning_rate_init", np.random.choice, learning_rate_init_list)
toolbox.register("attr_solver", np.random.choice, solver_list)

# 个体定义
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_activation, toolbox.attr_hidden_layer_sizes, toolbox.attr_learning_rate_init, toolbox.attr_solver), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 更新评估函数
def evalMLP(individual):
    activation = individual[0]
    hidden_layer_sizes = individual[1]
    learning_rate_init = individual[2]
    solver = individual[3]

    # 数据划分，random_state设置为0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 模型定义，random_state设置为0
    model = MLPRegressor(
        activation=activation,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        solver=solver,
        random_state=0,
        max_iter=2000  # 增加迭代次数以保证收敛
    )

    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    r2_train = r2_score(y_train, predictions_train)
    r2_test = r2_score(y_test, predictions_test)

    return (1 * r2_train + 0 * r2_test,)


toolbox.register("evaluate", evalMLP)

# 定义交叉和变异
toolbox.register("mate", tools.cxTwoPoint)

# 针对数值型参数进行高斯变异
def customMutate(individual):
    # 对 hidden_layer_sizes 和 learning_rate_init 进行高斯变异
    if isinstance(individual[1], tuple):  # hidden_layer_sizes
        hidden_layer_sizes_mut = np.random.choice(hidden_layer_sizes_list)
        individual[1] = hidden_layer_sizes_mut
    if isinstance(individual[2], float):  # learning_rate_init
        learning_rate_init_mut = np.random.uniform(0.001, 0.1)
        individual[2] = learning_rate_init_mut
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
    best_activation = best_individual[0]
    best_hidden_layer_sizes = best_individual[1]
    best_learning_rate_init = best_individual[2]
    best_solver = best_individual[3]
    print("最优超参数：activation={}, hidden_layer_sizes={}, learning_rate_init={}, solver={}".format(
        best_activation, best_hidden_layer_sizes, best_learning_rate_init, best_solver))

    # 重新计算最优模型的R2分数
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = MLPRegressor(activation=best_activation,
                         hidden_layer_sizes=best_hidden_layer_sizes,
                         learning_rate_init=best_learning_rate_init,
                         solver=best_solver,
                         random_state=0,
                         max_iter=2000)
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    weighted_r2 = 0.5 * r2_train + 0.5 * r2_test
    print("训练集 R2: {:.3f}, 测试集 R2: {:.3f}, 加权平均 R2: {:.3f}".format(r2_train, r2_test, weighted_r2))
