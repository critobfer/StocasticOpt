from geopy.distance import geodesic
import logging
import auxiliar_lib.optimization_problem as op
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import auxiliar_lib.ML_models as models

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def predict_demand(nodeData, demandData, realDemand, method):
    np.random.seed(100513471)
    # CLEAN THE DATE
    demandData = demandData.drop(['Date', 'Year'], axis=1)
    realDemand = realDemand.drop(['Date', 'Year'], axis=1)
    
    # DATA GENERATION
    points_ids = nodeData['codnode'].values
    n = len(points_ids)
    latitudes = []
    longitudes = []
    d = []
    n_train = []
    model_resutls = []
    i = 0
    for codnode in points_ids:
        node = nodeData[nodeData['codnode'] == codnode] 
        logger.info('We generate data from point ' + str(i))
        lat = node['latitude'].values[0]
        latitudes.append(lat)
        lon = node['longitude'].values[0]
        longitudes.append(lon)
        # We prepare the data
        demandNode = demandData[demandData['codnode'] == codnode]
        realDemandNode = realDemand[realDemand['codnode'] == codnode]
        X_train = demandNode.drop(['Pallets', 'codnode'], axis=1)
        y_train = demandNode['Pallets']
        X_test = realDemandNode.drop(['Pallets', 'codnode'], axis=1)
        if method == 'Linear Regression':
            result = models.linear_regresion(X_train=X_train, X_test=X_test, y_train=y_train)
        elif method == 'Random Forest':
            result = models.random_forest(X_train=X_train, X_test=X_test, y_train=y_train)
        elif method == 'SVR':
            result = models.svm(X_train=X_train, X_test=X_test, y_train=y_train)
        elif method == 'Neural Network':
            result = models.neural_network(X_train=X_train, X_test=X_test, y_train=y_train)
        elif method == 'XGBoosting':
            result = models.xgboost_lgbm(X_train=X_train, X_test=X_test, y_train=y_train)
        elif method == 'Lasso':
            result = models.lasso_regression(X_train=X_train, X_test=X_test, y_train=y_train)
        elif method == 'Ridge':
            result = models.ridge_regression(X_train=X_train, X_test=X_test, y_train=y_train)

        d.append(result['prediction'])
        n_train.append(len(X_train))
        model_resutls.append(result)
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

    return points_ids, c, d, D, latitudes, longitudes, n_train, model_resutls

def execute(option, nodeData, demandData, realDemand):
    num_nodos = len(nodeData)
    codnodes, c, d, D, latitudes, longitudes, n_train, model_resutls = predict_demand(nodeData, demandData, realDemand, option)
    model, results = op.prize_collecting_TSP(num_nodos, c, d, D)
    real_d = realDemand['Pallets'].values
    x_sol, y_sol, u_sol, capacity_used, opt_value, total_distance = op.feed_solution_variables(model, num_nodos, real_d, c)
    codnodes_achived = [codnodes[i] for i in range(num_nodos) if y_sol[i] == 1]
    tour_coords = op.get_tour_cord(x_sol, latitudes, longitudes, num_nodos)
    # Conpute metrics:
    mse = mean_squared_error(real_d, d)
    r2 = r2_score(real_d, d)
    mae = mean_absolute_error(real_d, d)

    result = {
        'codnodes_selected': codnodes,
        'num_nodes':len(codnodes),
        'codnodes_visited':codnodes_achived,
        'num_visited':len(codnodes_achived),
        'total_capacity': D,
        'capacity_used': capacity_used,
        'total_distance': total_distance,
        'nodes_demand': real_d,
        'nodes_predicted_demand': d,
        'n_train': n_train,
        'model_result': model_resutls,
        'MSE':mse,
        'R2':r2,
        'MAE':mae,
        'distance_matrix': c,
        'optimum_value': opt_value,
        'tour_coords': tour_coords,
        'nodeDataSelected': nodeData,
        'demandDataSelected': demandData,
        'method': option,
        'info': '_option_'+str(option)
    }

    return result