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

def prize_collecting_TSP_multiscenario(n, c, d, D, num_scenarios, probabilities, method: str, alpha:float = 0.8):
    opt = SolverFactory("gurobi")

    model = ConcreteModel()

    model.N = RangeSet(1, n) 
    model.N_reduced = RangeSet(2, n)
    model.S = RangeSet(1, num_scenarios)

    # The next line declares a variable indexed by the set points
    model.x = Var(model.N, model.N, domain=Binary, bounds=(0, 1))
    model.u = Var(model.N, domain=NonNegativeReals, bounds=(1, n)) 
    model.y = Var(model.N, domain=Binary, bounds=(0, 1))

    # Definition of the objective function
    if method == 'Maximum expectation':
        '''In this approach, one seeks to maximise the expected or average value of the outcome under uncertainty. 
        In the context of a stochastic optimisation problem, this means that the objective function is defined to 
        maximise the weighted sum of the outcomes of each scenario, where the weights are the probabilities of 
        occurrence of the scenarios. This approach is appropriate when the main objective is to obtain the best 
        possible average outcome, regardless of the variability in the outcomes.'''
        def obj_expression(model): 
            return sum(probabilities[s] * (sum(model.x[i, j] * c[i - 1][j - 1] for j in model.N) +
                                        (1 - model.y[i]) * d[s][i - 1]) for i in model.N for s in range(num_scenarios))
        model.OBJ = Objective(rule=obj_expression, sense=minimize) 
    elif method == 'Minimum variance':
        '''In this approach, one seeks to minimise the variability or dispersion in the outcomes under uncertainty. 
        To achieve this, the objective function can be defined to minimise the variance of the outcomes weighted by 
        the probabilities of the scenarios. Minimising variance implies reducing the dispersion of outcomes compared 
        to the expectation maximisation approach. This is useful when one wants to minimise uncertainty or variability 
        in the outcomes.'''
        def obj_expression(model): # TODO: Ya no es lineal
            mean = sum(probabilities[s] * (sum(model.x[i, j] * c[i - 1][j - 1] for j in model.N) +
                                        (1 - model.y[i]) * d[s][i - 1]) for i in model.N for s in range(num_scenarios))
            var = sum(probabilities[s] * (sum(model.x[i, j] * c[i - 1][j - 1] for j in model.N) +
                                (1 - model.y[i]) * d[s][i - 1] - mean) ** 2 for i in model.N for s in range(num_scenarios))
            return var
        model.OBJ = Objective(rule=obj_expression, sense=minimize) 
    elif method == 'Conditional Value at Risk (CVaR)':
        '''Risk aversion implies a preference for more certain outcomes over more uncertain outcomes, 
        even if the uncertain outcomes have a higher profit potential. In the context of a stochastic 
        optimisation problem, this can be achieved by introducing a utility function that penalises 
        riskier outcomes. This utility function can be convex or concave, depending on the degree of risk aversion. 
        This approach is useful when one wants to take into account the risk attitude of the decision-maker.'''
        model.eta = Var(model.S, domain=NonNegativeReals) #cvar auxiliary variable
        model.PI = Var(model.S) #cvar auxiliary variable
        model.tita = Var() #cvar auxiliary variable

        #constraint CVaR
        def cvar_cons(model,s): 
            return -model.PI[s] + model.tita + model.eta[s] >= 0  
        model.cvar_cons_cv = Constraint(model.S, rule=cvar_cons)

        #definition of OF per scenario
        def of_cv_cons(model,s):
            return model.PI[s] - sum(sum(model.x[i,j]*c[i-1][j-1] for j in model.N) for i in model.N) - sum((1-model.y[i])*d[s-1][i-1] for i in model.N) == 0
        model.costcv_cons_cons = Constraint(model.S, rule=of_cv_cons)

        def obj_expression(model):
            return model.tita + (1/(1-alpha))*sum(probabilities[s-1]*model.eta[s] for s in model.S) 
        
        model.OBJ = Objective(rule=obj_expression, sense=minimize) 
    elif method == 'Worst Case Analysis':
        model.t = Var() #worst case cost scenario

        #definition of OF per scenario
        def worst_case_cons(model,s):
            return sum(sum(model.x[i,j]*c[i-1][j-1] for j in model.N) for i in model.N) + sum((1-model.y[i])*d[s-1][i-1] for i in model.N) <= model.t
        model.costcv_cons_cons = Constraint(model.S, rule=worst_case_cons)

        def obj_expression(model): 
            return model.t
        model.OBJ = Objective(rule=obj_expression, sense=minimize) 

    # Only once from i
    def max_once_from_i(model, i): 
        return sum(model.x[i, j] for j in model.N) <= 1
    model.max_once_from_i_Constraint = Constraint(model.N, rule=max_once_from_i)

    # Only once from i
    def visited_or_not(model, i): 
        return sum(model.x[i, j] for j in model.N) == sum(model.x[j, i] for j in model.N)
    model.visited_or_not_Constraint = Constraint(model.N, rule=visited_or_not)

    # Only once to j
    def max_once_to_j(model, j): 
        return sum(model.x[i, j] for i in model.N) <= 1
    model.max_once_to_j_Constraint = Constraint(model.N, rule=max_once_to_j)

    # No subtours
    def no_sub_tours(model, i, j): 
        return model.u[i] - n * (1 - model.x[i, j]) <= model.u[j] - 1
    model.no_sub_tours_Constraint = Constraint(model.N, model.N_reduced, rule=no_sub_tours)

    # Visited constraint
    def visited(model, i): 
        return sum(model.x[i, j] for j in model.N) == model.y[i]
    model.visited_Constraint = Constraint(model.N, rule=visited)

    # Max capacity constraint
    def capacity(model, i, s): 
        return sum(model.y[i] * d[s][i - 1] for i in model.N) <= D
    model.capacity_Constraint = Constraint(model.N, range(num_scenarios), rule=capacity)

    start = time.time()
    results = opt.solve(model, tee=True)
    end  = time.time()
    logger.info('######################################################')
    logger.info('With ' + str(n) + ' points: '+ str(end-start) + 's')
    logger.info('######################################################')

    return model, results

def feed_solution_variables(model, n, d, c):
    capacity_used = 0
    x_sol = np.zeros((n,n))
    u_sol = np.zeros((n))
    y_sol = np.zeros((n))
    total_cost = 0 # In our case is distance
    for i in range(0, n):
        y_sol[i] = model.y[i+1].value
        u_sol[i] = model.u[i+1].value
        if y_sol[i]:
            capacity_used += d[i]
        for j in range(0,n):
            x_sol[i,j]=model.x[i+1,j+1].value
            if x_sol[i,j] == 1:
                total_cost += c[i][j]

    opt_value = model.OBJ()

    for i in range(0, n):
        u_sol[i] = model.u[i+1].value
        logger.info('The Node ' + str(i+1) + ' is the ' + str(int(u_sol[i])) + 'th point visited')

    logger.info('The minimum travel cost is ' + str(opt_value))

    num_dec_var = sum(1 for _ in model.component_data_objects(Var))
    num_cons = sum(1 for _ in model.component_data_objects(Constraint))

    logger.info('We have a total of' + str(num_dec_var) + 'decision varibales')
    logger.info('We have a total of'+ str(num_cons) + 'constraints')

    return x_sol, y_sol, u_sol, capacity_used, opt_value, total_cost

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
