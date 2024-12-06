from multiprocessing import Pool, cpu_count
import random
import math
import csv
import matplotlib.pyplot as plt
import functools
import time
vis = dict()
def parse_tsp_file(file_path):
    cities = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            cities.append((float(row[0]), float(row[1])))
    return cities

cities = parse_tsp_file('large.csv')
sz = 20

@functools.cache
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def crossover(parent1, parent2, cities=cities):
    vis = [0 for _ in range(len(parent1))]
    p, q = 0, 0
    child = []
    child.append(random.randint(0, len(cities) - 1))
    vis[child[0]] = 1
    while len(child) < len(parent1) and p < len(parent1) and q < len(parent2):
        while p < len(parent1) and vis[parent1[p]] == 1:
            p += 1
        while q < len(parent2) and vis[parent2[q]] == 1:
            q += 1
        if p >= len(parent1) or q >= len(parent2):
            break
        a = distance(cities[parent1[p]], cities[child[-1]])
        b = distance(cities[parent2[q]], cities[child[-1]])
        if a < b:
            child.append(parent1[p])
            vis[parent1[p]] = 1
            p += 1
        else:
            child.append(parent2[q])
            vis[parent2[q]] = 1
            q += 1

    if len(child) < len(parent1):
        while p < len(parent1):
            if vis[parent1[p]] != 1:
                child.append(parent1[p])
                vis[parent1[p]] = 1
            p += 1
    if len(child) < len(parent2):
        while q < len(parent2):
            if vis[parent2[q]] != 1:
                child.append(parent2[q])
                vis[parent2[q]] = 1
            q += 1

    return child


# def crossover(parent1, parent2):
#     child = parent1[:len(parent1) // 2]
#     child += [gene for gene in parent2 if gene not in child]
#     return child
# 貪婪算法
def greedy_algorithm(cities , i):
    visited = [i]  # 從第一個城市開始
    unvisited = list(range(0, len(cities)))
    unvisited[i] = 0
    current_city = i
    total_distance = 0

    while unvisited:
        # 找到最近的城市
        nearest_city = min(unvisited, key=lambda x: distance(cities[current_city], cities[x]))
        total_distance += distance(cities[current_city], cities[nearest_city])
        visited.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    # 回到起始城市
    total_distance += distance(cities[current_city], cities[0])
    visited.append(i)

    return visited, total_distance
# 適應度計算
def fitness(route, cities):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1))

# 初始化種群
def initialize_population(init_arr , pop_size , num_cities):
    # ret = []
    # for i in range(len(cities)):
    #     ret.append(greedy_algorithm(cities=cities ,i=i)[0])
    # return ret
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]
    # return [init_arr for _ in range(pop_size)]

# 選擇操作
def select_parents(population, cities):
    population.sort(key=lambda x: fitness(x, cities))
    return population[:sz]  # 選擇最佳的20%作為父代

# 變異操作
def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

init_arr = []
gen = 2
# 生成子代的函數
def generate_child(args):
    parent1, parent2, mutation_rate, cities = args
    child = mutate(crossover(parent1, parent2), mutation_rate)
    child.append(child[0])  # 閉合路線
    
    # 將child轉換為字符串列表
    now = ', '.join(map(str, child))
    while now in vis:
        # 如果child已經存在，重新生成
        child = random.shuffle(child[0 : len(child) - 1])
        # child = mutate(crossover(parent1, parent2), mutation_rate)
        child.append(child[0])
        now = ', '.join(map(str, child))
    
    # 將child添加到vis中
    vis[now] = 1
    return child

def close_route(route):
    route.append(route[0])
    now = ', '.join(map(str, route))
    vis[now] = 1
    return route


def record_route(route):
    route = ', '.join(map(str, route))
    vis[route] = 1
    return

def genetic_algorithm(cities, pop_size=10, generations=10, mutation_rate=0.01):
    num_cities = len(cities)  # 獲取城市的數量
    population = initialize_population(pop_size, pop_size=pop_size, num_cities=num_cities)  # 初始化種群
    with Pool(processes=cpu_count()) as pool:
        population = pool.map(close_route, population)  # 使用多處理優化種群中的路線
    best_fitness_history = []  # 紀錄每一代的最佳適應度值
    avg_fitness_history = []   # 紀錄每一代的平均適應度值

    for gen in range(generations):
        print(f'gen = {gen}')
        gen += 1  # 更新代數
        fitness_values = evaluate_population(population, cities)  # 評估當前種群中每條路線的適應度
        best_fitness = min(fitness_values)  # 獲取最佳適應度（最短距離）
        avg_fitness = sum(fitness_values) / len(fitness_values)  # 計算平均適應度
        best_fitness_history.append(best_fitness)  # 將最佳適應度加入歷史紀錄
        avg_fitness_history.append(avg_fitness)    # 將平均適應度加入歷史紀錄

        parents = select_parents(population, cities)  # 選擇父代進行交配
        # 使用多處理生成子代
        with Pool(processes=cpu_count()) as pool:
            new_population = parents + pool.map(
                generate_child,
                [
                    (
                        random.choice(parents),    # 隨機選擇父本
                        random.choice(parents),    # 隨機選擇母本
                        mutation_rate,             # 突變率
                        cities                     # 城市列表
                    ) for _ in range(pop_size - len(parents))
                ]
            )
        population = new_population  # 更新種群為新一代

    best_route = min(population, key=lambda x: fitness(x, cities))  # 從最終種群中獲取最佳路線
    return best_route, fitness(best_route, cities), best_fitness_history, avg_fitness_history

def evaluate_population(population, cities):
    with Pool() as pool:
        fitness_values = pool.starmap(fitness, [(route, cities) for route in population])  # 計算每條路線的適應度
    return fitness_values

def main():
    # 主函數入口
    pass  # 這裡可以填入您的主程式碼

if __name__ == "__main__":
    main()
