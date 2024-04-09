# StocasticOpt

This repository presents a streamlit application to experiment with various stochastic optimisation methods. 
We work with a Pize Collecting Travel Sales Man (PCTSP) where the penalty is the undelivered demand. 

You must choose the day you want to plan and the maximum number of clients/nodes.

Then you can choose between the methods:
--
* Deterministic: Solve the problem using the demands of the day.
* MultiScenario: It generate some scenarios with a lognormal taking into account the mean and the variance of the past registers, then we use a stocastic optimization problem focus on:
    - Maximum expectation
    - CVaR
    - Worst Scenario 
* Machine Learning: It predict the current demand with any of the following models (taking into account the covariables), then it execute the deterministic optimization problem with these predicted demands.
    - Linear Regression
    - Lasso
    - Ridge
    - Random Forest
    - SVR
    - Neural Network
    - XGBoosting
* KNN MultiScenario: In this case we generate scenarios but in a smarter way. We take as scenario the K Nearent Neighbour, taking into account the covariables. Then we use a stocastic optimization problem focus on: 
    - Maximum expectation
    - CVaR
    - Worst Scenario 

# Structure of the repository

# Set up
* You have to install the requirements.
* Then you have to set the environ variables with your two HERE API KEYS in a ``.env`` file as follow. 

``
API_KEY_1='apikey1'
API_KEY_2='apikey2'
``

* Execute ``main.py``
* To view this Streamlit app on a browser, you have to run a command like this  ``streamlit run c:/Users/.../StocasticOpt/main.py`` that you will see in a warning python after run ``main.py``
