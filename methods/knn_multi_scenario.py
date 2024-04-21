from geopy.distance import geodesic
import logging
import auxiliar_lib.optimization_problem as op
import auxiliar_lib.ML_models as models

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_data_scenarios(k, nodeData, demandData, realDemand):
    # CLEAN THE DATE
    demandData = demandData.drop(['Date', 'Year'], axis=1)
    realDemand = realDemand.drop(['Date', 'Year'], axis=1)
    # DATA GENERATION
    points_ids = nodeData['codnode'].values
    n = len(points_ids)
    latitudes = []
    longitudes = []
    d = [[] for _ in range(k)]
    min_max_dist = []
    i = 0
    for codnode in points_ids:
        node = nodeData[nodeData['codnode'] == codnode] 
        logger.info('We generate data from point ' + str(i))
        lat = node['latitude'].values[0]
        latitudes.append(lat)
        lon = node['longitude'].values[0]
        longitudes.append(lon)
        # We generate m scenarios
        demandNode = demandData[demandData['codnode'] == codnode]
        realDemandNode = realDemand[realDemand['codnode'] == codnode]
        X_train = demandNode.drop(['Pallets', 'codnode'], axis=1)
        y_train = demandNode['Pallets']
        X_test = realDemandNode.drop(['Pallets', 'codnode'], axis=1)
        neighbors_demand_values, min_dist, max_dist = models.get_knn_demand(k, X_train, X_test, y_train)
        for i in range(k):
            d[i].append(neighbors_demand_values[i])
        min_max_dist.append([min_dist, max_dist])
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

    return points_ids, c, d, D, latitudes, longitudes, min_max_dist

def execute(k, option, nodeData, demandData, realDemand, alpha):
    num_nodos = len(nodeData)
    codnodes, c, d, D, latitudes, longitudes, min_max_dist = generate_data_scenarios(k, nodeData, demandData, realDemand)
    # The first scenario has more probaility
    weight = [1 / i for i in range(1, k + 1)]
    total = sum(weight)
    prob = [w / total for w in weight] # To sum up 1

    model, results = op.prize_collecting_TSP_multiscenario(num_nodos, c, d, D, k, prob, option, alpha=alpha)
    real_d = realDemand['Pallets'].values
    x_sol, y_sol, u_sol, capacity_used, opt_value, total_distance = op.feed_solution_variables(model, num_nodos, real_d, c)
    codnodes_achived = [codnodes[i] for i in range(num_nodos) if y_sol[i] == 1]
    tour_coords = op.get_tour_cord(x_sol, latitudes, longitudes, num_nodos)
    if option == 'Conditional Value at Risk (CVaR)':
        extra_info = '_alpha_'+str(alpha)
    else:
        extra_info = ''
    result = {
        'codnodes_selected': codnodes,
        'num_nodes':len(codnodes),
        'codnodes_visited':codnodes_achived,
        'num_visited':len(codnodes_achived),
        'total_capacity': D,
        'capacity_used': capacity_used,
        'total_distance': total_distance,
        'nodes_demand': real_d, 
        'k':k,
        'min_max_dist':min_max_dist,
        'nodes_demand_multiscenario': d,
        'distance_matrix': c,
        'optimum_value': opt_value,
        'tour_coords': tour_coords,
        'nodeDataSelected': nodeData,
        'demandDataSelected': demandData,
        'info': '_k_'+str(k)+'_option_'+str(option)+extra_info
    }

    return result