import random
# !pip install geopy
from geopy.distance import geodesic
from pyomo.environ import *
from pyomo.opt import SolverFactory
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_data():
    random.seed(100513471)

    # DATA GENERATION
    n = 50 # Number of points

    # We generate n point in Andalucía
    min_lat, max_lat = 36.0, 38.5
    min_lon, max_lon = -7.5, -3.0
    latitudes = []
    longitudes = []
    for i in range(n):
        logger.info('We generate data from point ' + str(i))
        lat = round(random.uniform(min_lat, max_lat), 6)
        latitudes.append(lat)
        lon = round(random.uniform(min_lon, max_lon), 6)
        longitudes.append(lon)

    # Cost matrix, in this case distance
    D = 40*n
    c = [[0] * n for _ in range(n)]
    d = [random.randint(0,100) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            coord_punto_1 = (latitudes[i], longitudes[i])
            coord_punto_2 = (latitudes[j], longitudes[j])
            distancia = geodesic(coord_punto_1, coord_punto_2).kilometers
            c[i][j] = distancia
        c[i][i] = 1000000

    return n, c, d, D, latitudes, longitudes

def read_data(num_nodos, nodeData, demandData):
    random.seed(100513471)

    # DATA GENERATION
    n = num_nodos # Number of points

    points_ids = random.sample(range(1, len(nodeData) + 1), n)

    latitudes = []
    longitudes = []
    d = [random.randint(0,100) for _ in range(n)]
    i = 0
    for codnode in points_ids:
        node = nodeData[nodeData['codnode'] == codnode] 
        logger.info('We generate data from point ' + str(i))
        lat = node['latitude'].values[0]
        latitudes.append(lat)
        lon = node['longitude'].values[0]
        longitudes.append(lon)
        demand = demandData[demandData['codnode'] == 1]['Pallets'].mean()
        d.append(demand)
        i+=1

    # Cost matrix, in this case distance
    D = 40*n
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            coord_punto_1 = (latitudes[i], longitudes[i])
            coord_punto_2 = (latitudes[j], longitudes[j])
            distancia = geodesic(coord_punto_1, coord_punto_2).kilometers
            c[i][j] = distancia
        c[i][i] = 1000000

    return points_ids, c, d, D, latitudes, longitudes

def prize_collecting_TSP(n, c, d, D):
    opt = SolverFactory("gurobi")

    model = ConcreteModel()

    model.N = RangeSet(1,n) 
    model.N_reduced = RangeSet(2,n) 

    # The next line declares a variable indexed by the set points
    model.x = Var(model.N, model.N, domain=Binary, bounds = (0,1))
    model.u = Var(model.N, domain=NonNegativeReals, bounds = (1,n)) 
    model.y = Var(model.N, domain=Binary, bounds = (0,1))

    #Definition of the objective function
    def obj_expression(model): 
        return sum(sum(model.x[i,j]*c[i-1][j-1] for j in model.N) for i in model.N) + sum((1-model.y[i])*d[i-1] for i in model.N)
    model.OBJ = Objective(rule=obj_expression, sense=minimize) 

    # Only once from i
    def max_once_from_i(model, i): 
        return sum(model.x[i,j] for j in model.N) <= 1
    model.max_once_from_i_Constraint = Constraint(model.N, rule=max_once_from_i)

    # Only once from i
    def visited_or_not(model, i): 
        return sum(model.x[i,j] for j in model.N) == sum(model.x[j,i] for j in model.N)
    model.visited_or_not_Constraint = Constraint(model.N, rule=visited_or_not)

    # Only once to j
    def max_once_to_j(model, j): 
        return sum(model.x[i,j] for i in model.N) <= 1
    model.max_once_to_j_Constraint = Constraint(model.N, rule=max_once_to_j)


    #No subtours
    def no_sub_tours(model,i, j): 
        return model.u[i] - n*(1-model.x[i,j]) <= model.u[j] - 1
    model.no_sub_tours_Constraint = Constraint(model.N, model.N_reduced, rule=no_sub_tours)

    #Visited constraint
    def visited(model, i): 
        return sum(model.x[i,j] for j in model.N) == model.y[i]
    model.visited_Constraint = Constraint(model.N, rule=visited)

    #Max capacity constraint
    def capacity(model, i): 
        return sum(model.y[i]*d[i-1] for i in model.N) <=D
    model.capacity_Constraint = Constraint(model.N, rule=capacity)

    start = time.time()
    results = opt.solve(model,tee=True)
    end  = time.time()
    logger.info('######################################################')
    logger.info('With ' + str(n) + ' points: '+ str(end-start) + 's')
    logger.info('######################################################')


    return model, results

def print_solution(model, n, d):
    capacity_used = 0
    x_sol = np.zeros((n,n))
    u_sol = np.zeros((n))
    y_sol = np.zeros((n))
    for i in range(0, n):
        y_sol[i] = model.y[i+1].value
        u_sol[i] = model.u[i+1].value
        if y_sol[i]:
            capacity_used += d[i]
        for j in range(0,n):
            x_sol[i,j]=model.x[i+1,j+1].value

    opt_value = model.OBJ()

    for i in range(0, n):
        u_sol[i] = model.u[i+1].value
        logger.info('The Node ' + str(i+1) + ' is the ' + str(int(u_sol[i])) + 'th point visited')

    logger.info('The minimum travel cost is ' + str(opt_value))

    num_dec_var = sum(1 for _ in model.component_data_objects(Var))
    num_cons = sum(1 for _ in model.component_data_objects(Constraint))

    logger.info('We have a total of' + str(num_dec_var) + 'decision varibales')
    logger.info('We have a total of'+ str(num_cons) + 'constraints')

    return x_sol, y_sol, u_sol, capacity_used, opt_value

def graph_solution(x_sol, y_sol, u_sol, capacity_used, latitudes, longitudes, n):
    current_node = 0
    tour_nodes = ['NOD_' + str(current_node+1)]
    tour = [current_node]


    while True:
        next_node = np.argmax(x_sol[current_node])
        if next_node == 0:
            break  # Hemos vuelto al nodo inicial, terminamos el recorrido
        tour_nodes.append('NOD_' + str(next_node+1))
        tour.append(next_node)
        current_node = next_node

    coords = np.array([[latitudes[i], longitudes[i]] for i in range(n)])

    # Add initial point at the end
    tour.append(tour[0])

    # Points coordinates
    tour_coords = coords[tour]

    # Plot all coordinates
    # plt.scatter(coords[:, 0], coords[:, 1], color='gray')
    # # Add cost and demand label at each point
    # for i in range(n):
    #     plt.text(coords[i, 0] + 0.03, coords[i, 1], f'{d[i]}', color='black', fontsize=8, fontweight='bold')

    # plt.text(36.1, -3.5, 'used:' + str(capacity_used), color='black', fontsize=15, fontweight='bold')
    # plt.text(36.1, -3.8, 'total:' + str(D), color='black', fontsize=15, fontweight='bold')

    # plt.plot(tour_coords[:, 0], tour_coords[:, 1], marker='o')
    # # Initial point
    # plt.plot(coords[tour[0], 0], coords[tour[0], 1], marker='x', color='red', markersize=10)



    # plt.title('PC TSP tour (starting in Node 1)')
    # plt.xlabel('Latitud')
    # plt.ylabel('Longitud')
    # plt.show()
    return tour_coords


def solve_problem(method, num_nodos, nodeData, demandData):
    codnodes, c, d, D, latitudes, longitudes= read_data(num_nodos, nodeData, demandData)
    model, results = prize_collecting_TSP(num_nodos, c, d, D)
    x_sol, y_sol, u_sol, capacity_used, opt_value = print_solution(model, num_nodos, d)
    codnodes_achived = [codnodes[i] for i in range(num_nodos) if y_sol[i] == 1]
    tour_coords = graph_solution(x_sol, y_sol, u_sol, capacity_used, latitudes, longitudes, num_nodos)

    result = {
        'codnodes_selected': codnodes,
        'num_nodes':len(codnodes),
        'codnodes_visited':codnodes_achived,
        'num_visited':len(codnodes_achived),
        'total_capacity': D,
        'capacity_used': capacity_used,
        'nodes_demand': d,
        'distance_matrix': c,
        'optimum_value': opt_value,
        'tour_coords': tour_coords
    }

    return result

# if __name__=="__main__":
#     # n, c, d, D, latitudes, longitudes = generate_data()
#     n, c, d, D, latitudes, longitudes = read_data(20)
#     model, results = prize_collecting_TSP(n, c, d, D)
#     x_sol, y_sol, u_sol, capacity_used = print_solution(model)
#     graph_solution(x_sol, y_sol, u_sol, capacity_used, latitudes, longitudes)