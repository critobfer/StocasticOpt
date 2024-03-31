import random
from geopy.distance import geodesic
import logging
import optimization_problem as op
import numpy as np
from scipy.stats import lognorm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_data_scenarios(num_nodos, num_scenarios, nodeData, demandData):
    random.seed(100513471)

    # DATA GENERATION
    n = num_nodos # Number of points

    points_ids = random.sample(range(1, len(nodeData) + 1), n)

    latitudes = []
    longitudes = []
    d = [[] for _ in range(num_scenarios)]
    params_simulation = []
    i = 0
    for codnode in points_ids:
        node = nodeData[nodeData['codnode'] == codnode] 
        logger.info('We generate data from point ' + str(i))
        lat = node['latitude'].values[0]
        latitudes.append(lat)
        lon = node['longitude'].values[0]
        longitudes.append(lon)
        # We generate m scenarios
        demand_node = demandData[demandData['codnode'] == codnode]['Pallets'].values
        mu = np.mean(demand_node)
        sigma = np.std(demand_node)
        params_simulation.append((mu, sigma))
        # TODO: Revisar que no da negativo==> Usamos la lognormal
        # https://www.probabilidadyestadistica.net/distribucion-lognormal/#grafica-de-la-distribucion-lognormal
        log_demand = np.log(demand_node)
        mu_log = np.mean(log_demand)
        sigma_log = np.std(log_demand)
        sample = np.random.normal(mu_log, sigma_log, num_scenarios)
        for s in range(num_scenarios):
            d[s].append(np.exp(sample[s]))
        i+=1

    # Cost matrix, in this case distance
    D = 150*n
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            coord_punto_1 = (latitudes[i], longitudes[i])
            coord_punto_2 = (latitudes[j], longitudes[j])
            distancia = geodesic(coord_punto_1, coord_punto_2).kilometers
            c[i][j] = distancia
        c[i][i] = 1000000

    return points_ids, c, d, D, latitudes, longitudes, params_simulation

def execute(num_nodos, num_scenarios, option, nodeData, demandData):
    codnodes, c, d, D, latitudes, longitudes, params_simulation= generate_data_scenarios(num_nodos, num_scenarios, nodeData, demandData)
    prob = [1/num_scenarios for _ in range(num_scenarios)]
    model, results = op.prize_collecting_TSP_multiscenario(num_nodos, c, d, D, num_scenarios, prob, option)
    x_sol, y_sol, u_sol, capacity_used, opt_value = op.feed_solution_variables_multiscenario(model, num_nodos, num_scenarios, d)
    codnodes_achived = [codnodes[i] for i in range(num_nodos) if y_sol[i] == 1]
    tour_coords = op.get_tour_cord(x_sol, latitudes, longitudes, num_nodos)
    d_array = np.array(d)
    result = {
        'codnodes_selected': codnodes,
        'num_nodes':len(codnodes),
        'codnodes_visited':codnodes_achived,
        'num_visited':len(codnodes_achived),
        'total_capacity': D,
        'capacity_used': capacity_used,
        'nodes_demand': [np.mean([d_array[s][i] for s in range(num_scenarios)]) for i in range(num_nodos)],
        'num_scenarios':num_scenarios,
        'nodes_demand_multiscenario': d,
        'params_simulations': params_simulation,
        'distance_matrix': c,
        'optimum_value': opt_value,
        'tour_coords': tour_coords
    }

    return result