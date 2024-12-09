from multiprocessing import Pool, cpu_count
import random
import math
import csv
# import matplotlib.pyplot as plt
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
    parent1 = parent1[:-1]
    parent2 = parent2[:-1]
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
greedy_ans = []
# 貪婪算法
def greedy_algorithm(cities , i):
    visited = [i]  # 從第一個城市開始
    # unvisited = list(range(0, len(cities)))
    unvisited = [j for j in range(len(cities))]
    unvisited.remove(i)
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
        child = list(range(len(cities)))
        random.shuffle(child)
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
# 遺傳算法的主要邏輯
def genetic_algorithm(cities, pop_size=10, generations=10, mutation_rate=0.01):
    num_cities = len(cities)
    population = initialize_population(pop_size, pop_size=pop_size , num_cities=num_cities)
    with Pool(processes=cpu_count()) as pool:
        population = pool.map(close_route, population)
    best_fitness_history = []
    avg_fitness_history = []
    for gen in range(generations):
        print(f'gen = {gen}')
        fitness_values = evaluate_population(population, cities)
        best_fitness = min(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        parents = select_parents(population, cities)
        with Pool(processes=cpu_count()) as pool:
            new_population = parents + pool.map(generate_child, [(random.choice(parents), random.choice(parents), mutation_rate, cities) for _ in range(pop_size - len(parents))])
        population = new_population

    best_route = min(population, key=lambda x: fitness(x, cities))
    return best_route, fitness(best_route, cities), best_fitness_history, avg_fitness_history

def evaluate_population(population, cities):
    with Pool() as pool:
        fitness_values = pool.starmap(fitness, [(route, cities) for route in population])
    return fitness_values


def main():
    with Pool(processes=cpu_count()) as pool:
        init_arr = pool.starmap(greedy_algorithm, [(cities , i) for i in range(len(cities))])

    greedy_ans = min(init_arr , key=lambda x: x[1])
    print(greedy_ans)
 

    pop = 1000
    gen = 10
    mutate = 0

    start_time = time.time()
    best_route, best_distance, best_fitness_history, avg_fitness_history = genetic_algorithm(pop_size=pop , generations=gen , mutation_rate=mutate ,cities=cities)
    end_time = time.time()
    print(f"Best Route: {best_route}")
    print(f"Total Distance: {best_distance}")
    tim = end_time - start_time
    hour = tim // 60 // 60
    minute = (tim - hour * 60 * 60) // 60
    sec = tim - (minute * 60) - (hour * 60 * 60)
    print(f'execute time: {hour} h {minute} m {sec} sec')
    plt.plot(range(len(best_fitness_history)), best_fitness_history, label=f"{sz}parent_{pop}pop_{gen}gen_{mutate}mutationRate" ,color='green')


    mutate = 0.01
    start_time = time.time()
    best_route, best_distance, best_fitness_history, avg_fitness_history = genetic_algorithm(pop_size=pop , generations=gen , mutation_rate=mutate, cities=cities)
    end_time = time.time()
    print(f"Best Route: {best_route}")
    print(f"Total Distance: {best_distance}")
    tim = end_time - start_time
    hour = tim // 60 // 60
    minute = (tim - hour * 60 * 60) // 60
    sec = tim - (minute * 60) - (hour * 60 * 60)
    print(f'execute time: {hour} h {minute} m {sec} sec')
    plt.plot(range(len(best_fitness_history)), best_fitness_history, label=f"{sz}parent_{pop}pop_{gen}gen_{mutate}mutationRate" ,color='blue')

    mutate = 0.1
    start_time = time.time()
    best_route, best_distance, best_fitness_history, avg_fitness_history = genetic_algorithm(pop_size=pop , generations=gen , mutation_rate=mutate ,cities=cities)
    print(f"Best Route: {best_route}")
    print(f"Total Distance: {best_distance}")
    end_time = time.time()
    tim = end_time - start_time
    hour = tim // 60 // 60
    minute = (tim - hour * 60 * 60) // 60
    sec = tim - (minute * 60) - (hour * 60 * 60)
    print(f'execute time: {hour} h {minute} m {sec} sec')
    # 顯示最終的進化圖表
    plt.ioff()
    plt.plot(range(len(best_fitness_history)), best_fitness_history, label=f"{sz}parent_{pop}pop_{gen}gen_{mutate}mutationRate" ,color='red')
    plt.title('Fitness Progression in Genetic Algorithm')
    plt.xlabel('Generation')
    plt.ylabel('Route Distance')
    # plt.ylim(0, 550)  # 設定 y 軸範圍
    plt.legend()
    plt.savefig('population.png')  # 保存圖表並設置背景透明
    plt.show()

if __name__ == "__main__":
    main()
