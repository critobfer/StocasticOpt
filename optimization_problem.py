from pyomo.environ import *
from pyomo.opt import SolverFactory
import logging
import time
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

def feed_solution_variables(model, n, d):
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

def get_tour_cord(x_sol, latitudes, longitudes, n):
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

    return tour_coords
