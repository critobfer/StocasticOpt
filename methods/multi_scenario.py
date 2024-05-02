from geopy.distance import geodesic
import logging
import auxiliar_lib.optimization_problem as op
import numpy as np
from scipy.stats import lognorm
import streamlit as st

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def generate_data_scenarios(num_scenarios, nodeData, demandData, capacity_per_client):
    np.random.seed(100513471)

    # DATA GENERATION
    points_ids = nodeData['codnode'].values
    n = len(points_ids)
    latitudes = []
    longitudes = []
    d = [[] for _ in range(num_scenarios)]
    params_simulation = []
    i = 0
    progress_text = "Generating scenarios for each node. Please wait."
    my_bar = st.progress(i, text=progress_text)
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
        # https://www.probabilidadyestadistica.net/distribucion-lognormal/#grafica-de-la-distribucion-lognormal
        log_demand = np.log(demand_node)
        mu_log = np.mean(log_demand)
        sigma_log = np.std(log_demand)
        sample = np.random.normal(mu_log, sigma_log, num_scenarios)
        for s in range(num_scenarios):
            d[s].append(np.exp(sample[s]))
        i+=1
        my_bar.progress(i/n, text=progress_text)

    # Cost matrix, in this case distance
    D = capacity_per_client*n
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            coord_punto_1 = (latitudes[i], longitudes[i])
            coord_punto_2 = (latitudes[j], longitudes[j])
            distancia = geodesic(coord_punto_1, coord_punto_2).kilometers
            c[i][j] = distancia
        c[i][i] = 1000000

    return points_ids, c, d, D, latitudes, longitudes, params_simulation

def get_scenarios_OF_value(model, num_scenarios):
    of_values = [None for _ in range(num_scenarios)]
    for s in range(num_scenarios):
        of_values[s] = model.PI[s+1].value
    return of_values

def execute(num_scenarios, option, nodeData, demandData, realDemand, alpha, capacity_per_client, cost_per_km , cost_per_no_del_demand):
    num_nodos = len(nodeData)
    st.write("Generating scenarios...")
    codnodes, c, d, D, latitudes, longitudes, params_simulation= generate_data_scenarios(num_scenarios, nodeData, demandData, capacity_per_client)
    prob = [1/num_scenarios for _ in range(num_scenarios)]
    st.write('Running optimization...')
    model, _ = op.prize_collecting_TSP_multiscenario(num_nodos, c, d, D, num_scenarios, prob, option, alpha, cost_per_km , cost_per_no_del_demand)
    real_d = realDemand['Pallets'].values
    of_values = get_scenarios_OF_value(model, num_scenarios)
    x_sol, y_sol, u_sol, capacity_used, opt_value, total_distance = op.feed_solution_variables(model, num_nodos, real_d, c)
    codnodes_achived = [codnodes[i] for i in range(num_nodos) if y_sol[i] == 1]
    tour_coords = op.get_tour_cord(x_sol, latitudes, longitudes, num_nodos)
    if option == 'Conditional Value at Risk (CVaR)':
        extra_info = '_alpha_'+str(alpha)
        title = ' with alpha '+str(alpha) 
    else:
        extra_info = ''
        title = ''
    result = {
        'codnodes_selected': codnodes,
        'num_nodes':len(codnodes),
        'codnodes_visited':codnodes_achived,
        'num_visited':len(codnodes_achived),
        'total_capacity': D,
        'capacity_used': capacity_used,
        'total_distance': total_distance,
        'nodes_demand': real_d, # [np.mean([d_array[s][i] for s in range(num_scenarios)]) for i in range(num_nodos)],
        'num_scenarios':num_scenarios,
        'nodes_demand_multiscenario': d,
        'objective_function_value_multiscenario': of_values,
        'params_simulations': params_simulation,
        'distance_matrix': c,
        'optimum_value': opt_value,
        'tour_coords': tour_coords,
        'nodeDataSelected': nodeData,
        'demandDataSelected': demandData,
        'info': '_numsc_'+str(num_scenarios)+'_option_'+str(option)+extra_info,
        'title':title
    }

    return result