import random
# !pip install geopy
from geopy.distance import geodesic
import logging
import optimization_problem as op

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

def execute(num_nodos, nodeData, demandData):
    codnodes, c, d, D, latitudes, longitudes= read_data(num_nodos, nodeData, demandData)
    model, results = op.prize_collecting_TSP(num_nodos, c, d, D)
    x_sol, y_sol, u_sol, capacity_used, opt_value = op.feed_solution_variables(model, num_nodos, d)
    codnodes_achived = [codnodes[i] for i in range(num_nodos) if y_sol[i] == 1]
    tour_coords = op.get_tour_cord(x_sol, y_sol, u_sol, capacity_used, latitudes, longitudes, num_nodos)

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