import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
import folium # https://folium.streamlit.app/
from streamlit_folium import st_folium 
from streamlit_extras.dataframe_explorer import dataframe_explorer 
import auxiliar_lib.here as here
import random
import methods.deterministic as det
import methods.multi_scenario as ms
import methods.knn_multi_scenario as kms
import methods.machine_learning as ml
import os

####################################################################################################
# AUXILIAR FUNCTIONS     ###########################################################################
####################################################################################################

def select_execution_data(demandData, nodeData, max_num_nodes, date, path):
    random.seed(10051347)
    codnode_list = list(demandData[demandData['Date'] == date]['codnode'].values)
    number_nodes = len(codnode_list)
    if number_nodes > max_num_nodes:
        codnode_list = random.sample(codnode_list, max_num_nodes)
    nodeDataModel = nodeData[nodeData['codnode'].isin(codnode_list)]

    demandDataModel = demandData[(demandData['Date'] < date) & 
                                 (demandData['codnode'].isin(codnode_list))]
    realDemand= demandData[(demandData['Date'] == date) & (demandData['codnode']
                                                           .isin(codnode_list))]                       
    # Sort based on codnode_list
    nodeDataModel = nodeDataModel.sort_values(by='codnode')
    demandDataModel = demandDataModel.sort_values(by='codnode')
    realDemand = realDemand.sort_values(by='codnode')

    nodeDataModel.to_csv(os.path.join(path, "nodeDataSelected.csv"), index=False)
    demandDataModel.to_csv(os.path.join(path, "demandDataSelected.csv"), index=False)
    realDemand.to_csv(os.path.join(path, "realDemand.csv"), index=False)
    return nodeDataModel, demandDataModel, realDemand

def create_path(date):
    output_directory = os.path.join("output_files", date.strftime("%Y-%m-%d"))
    
    # Crear el directorio si no existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_directory

def add_path(path, method):
    output_directory = os.path.join(path, method)
    
    # Crear el directorio si no existe
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_directory

def write_to_result_file(result_file_path, data):
    with open(result_file_path, 'a') as file:
        file.write(data + "\n")
####################################################################################################
# AUXILIAR VARIABLES     ###########################################################################
####################################################################################################
month_dict = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
month_dict_reversed = {value: key for key, value in month_dict.items()}

####################################################################################################
# STREAMLIT CONFIGURATION    #######################################################################
####################################################################################################

st.set_option('deprecation.showPyplotGlobalUse', False)

# @st.cache_data #Takes you cache_resource arguments for machine learning 

# Configuring the Streamlit application
st.title('Traveling Salesman Problem under Demand Uncertainty :truck:')
st.markdown('This application uses several methods to solve the stochastic problem.')
st.divider()

####################################################################################################
# SIDE BAR CONFIGURATION ###########################################################################
####################################################################################################
st.sidebar.header('Configuration', divider='red')


###################################################################################################
# FILES                  ##########################################################################
###################################################################################################
# Select file from computer
# nodeData = st.sidebar.file_uploader(":open_file_folder: Node File", type=["csv"])
# if nodeData is not None:
#     nodeData = pd.read_csv(nodeData, encoding='latin-1', sep=';')
#     st.session_state["nodeData"] = nodeData
# demandData = st.sidebar.file_uploader(":open_file_folder: Demand File", type=["csv"])
# if demandData is not None:
#     demandData = pd.read_csv(demandData, encoding='latin-1', sep=';')
#     demandData['Date'] = pd.to_datetime(demandData['Date'])
#     st.session_state["demandData"] = demandData

# Get file from github
nodeData_github_link = "https://raw.githubusercontent.com/critobfer/StocasticOpt/main/data/nodeData.csv"
demandData_github_link = "https://raw.githubusercontent.com/critobfer/StocasticOpt/main/data/demandDataComplete.csv"

nodeData = pd.read_csv(nodeData_github_link, sep=';', encoding='latin-1') 
if nodeData is not None:
    st.session_state["nodeData"] = nodeData

demandData = pd.read_csv(demandData_github_link, sep=';', encoding='latin-1') 
if demandData is not None:
    demandData['Date'] = pd.to_datetime(demandData['Date'])

    st.session_state["demandData"] = demandData

###################################################################################################
# SELECT DATE            ##########################################################################
###################################################################################################
st.sidebar.subheader('Day we want to solve', divider='red')
if demandData is not None:
    year = st.sidebar.selectbox('Select the year', options=[2023, 2024])
    if year == 2023:
        months = [month_dict[i] for i in list(demandData[demandData['Year'] == year]
                                            ['Month'].sort_values()) if i >= 7]
    else:
        months = [month_dict[i] for i in list(demandData[demandData['Year'] == year]
                                    ['Month'].sort_values())]

    month_name = st.sidebar.select_slider('Select the month', options=months)
    month = month_dict_reversed[month_name]
    days = list(demandData[(demandData['Year'] == year) & (demandData['Month'] == month)]
                ['Day'].sort_values())
    day = st.sidebar.select_slider('Select the day', options=days)
    date = datetime(year, month, day)

###################################################################################################
# SELECT METHOD          ##########################################################################
###################################################################################################
st.sidebar.subheader('Method', divider='red')
# Drop-down controls and bars
method = st.sidebar.selectbox('Select Method', ['Deterministic', 'Multi-scenario', 'KNN Multi-scenario',
                                                'Machine Learning'])

###################################################################################################
# SELECT PARAMETERS     ###########################################################################
###################################################################################################
# Others parameters for the methods
st.sidebar.subheader('Parameters', divider='red')
if method == 'Multi-scenario':
    num_scenarios = st.sidebar.number_input('Number of Scenarios', min_value=1, step=1, value=30)
    ms_option = st.sidebar.selectbox('Options', ['Maximum expectation', 
                                                 'Conditional Value at Risk (CVaR)', 
                                                 'Worst Case Analysis'])
elif method == 'KNN Multi-scenario':
    k = st.sidebar.number_input('Choose the number of neighbour', min_value=1, max_value=40, step=1, value=20)
    ms_option = st.sidebar.selectbox('Options', ['Maximum expectation', 
                                                 'Conditional Value at Risk (CVaR)', 
                                                 'Worst Case Analysis'])
elif method == 'Machine Learning':
    ml_option = st.sidebar.selectbox('Machine Learning Options', ['Linear Regression', 'Lasso', 'Ridge',
                                                                  'Random Forest',
                                                                  'SVR', 'Neural Network', 'XGBoosting'])
max_num_nodes = st.sidebar.slider('Max number of points', min_value=3, max_value=40, value=15)

####################################################################################################
# EXECUTE METHODS        ###########################################################################
####################################################################################################
# Optional display of nodes
if st.sidebar.button('Solve', type='primary', use_container_width=True ):
    if "nodeData" not in st.session_state or "demandData" not in st.session_state:
        if "nodeData" not in st.session_state:
            st.warning('We need a node data file', icon="⚠️")
        if "demandData" not in st.session_state:
            st.warning('We need a demand data file', icon="⚠️")
        st.stop()

    path = create_path(date)
    nodeDataSelected, demandDataSelected, realDemand = select_execution_data(demandData, 
                                                                             nodeData, 
                                                                             max_num_nodes, 
                                                                             date, path)
    path_method = add_path(path, method)
    st.session_state["path_method"] = path_method

    # Mostrar el spinner
    if method == 'Deterministic':
        with st.spinner('Executing model'):
            result = det.execute(nodeData=nodeDataSelected, realDemand=realDemand, 
                                 demandData=demandDataSelected) 
        st.empty()
        st.session_state["result"] = result
    elif method == 'Multi-scenario':
        with st.spinner('Executing model'):
            result = ms.execute(num_scenarios=num_scenarios, option=ms_option, 
                                nodeData=nodeDataSelected, demandData=demandDataSelected, 
                                realDemand=realDemand) 
        st.empty()
        st.session_state["result"] = result
    elif method == 'KNN Multi-scenario':
        with st.spinner('Executing model'):
            result = kms.execute(k=k, option=ms_option,
                                nodeData=nodeDataSelected, demandData=demandDataSelected, 
                                realDemand=realDemand) 
        st.empty()
        st.session_state["result"] = result
    elif method == 'Machine Learning':

        with st.spinner('Executing model'):
            result = ml.execute(option=ml_option, nodeData=nodeDataSelected, 
                                demandData=demandDataSelected,
                                realDemand=realDemand) 
        st.empty()
        st.session_state["result"] = result
    else:
        st.warning('We are working on it', icon="🔧")
        st.stop()

####################################################################################################
# PRINT RESULTS          ###########################################################################
####################################################################################################
if "result" in st.session_state:
    result = st.session_state["result"]
    result_file_path = os.path.join(st.session_state["path_method"], "result" + result['info'] +".txt")

    nodeDataSelected = result['nodeDataSelected']
    demandDataSelected = result['demandDataSelected']
    
    st.header('Result:', divider='red')

    ###################################################################################################
    # OBJECTIVE FUNCTION    ###########################################################################
    ###################################################################################################
    # Show Objective Function
    st.subheader(f'**Objective Function:** {round(result['total_distance'] +
                                            np.sum(result['nodes_demand']) - 
                                                  result['capacity_used'] ,2)}')
    write_to_result_file(result_file_path, f'Objective Function: {round(result['total_distance'] +
                                            np.sum(result['nodes_demand']) - 
                                                  result['capacity_used'] ,2)}')

    col1, col2= st.columns(2)
    with col1:
        st.metric("Distance travelled", f'{round(result['total_distance'],2)}Km', "")
        write_to_result_file(result_file_path, f'Distance travelled: {round(result['total_distance'],2)}Km')
    with col2:
        st.metric("Undelivered demand", f'{round(np.sum(result['nodes_demand'])
                                               - result['capacity_used'] ,2)} Pallets', "")
        write_to_result_file(result_file_path, f'Undelivered demand: {round(np.sum(result['nodes_demand'])
                                               - result['capacity_used'] ,2)} Pallets')
    st.markdown(f'***Method Objective function:*** {round(result['optimum_value'],2)}')
    write_to_result_file(result_file_path, f'Method Objective function: {round(result['optimum_value'],2)}')

    ###################################################################################################
    # MAP                   ###########################################################################
    ###################################################################################################
    m = folium.Map(location=[nodeDataSelected['latitude'].values.mean(), 
                             nodeDataSelected['longitude'].values.mean()], zoom_start=11, 
                             tiles = "CartoDB Positron")
    for i in range(result['num_nodes']):
        popup_html = """
        <div style="width: 200px;">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
            <div style="margin-bottom: 10px;"> 
                <p style="display: inline-block; width: 15px;"><i class="fa fa-cubes" style="font-size: 14px;"></i></p>
                <p style="display: inline-block; margin-left: 5px;"><strong>Demand:</strong> {demand} </p>
            </div>
            <div> 
                <p style="display: inline-block; width: 15px;"><i class="fa fa-map-marker" style="font-size: 15px;"></i></p>
                <p style="display: inline-block; margin-left: 5px;"><strong>Node {codnode}:</strong> ({latitude}, {longitude})</p>
            </div>
        </div>
        """.format(demand=round(result['nodes_demand'][i], 2),
                    latitude=round(nodeDataSelected['latitude'].values[i], 2),
                    longitude=round(nodeDataSelected['longitude'].values[i], 2),
                    codnode=nodeDataSelected['codnode'].values[i])
        # Add Markers
        folium.Marker( [nodeDataSelected['latitude'].values[i], nodeDataSelected['longitude'].values[i]],
            tooltip=nodeDataSelected['Business'].values[i],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red')).add_to(m)
    # Add road paths
    if result['num_visited'] > 0:
        # We convert into a hasheable in order to use the session
        tour_coords = [tuple(coord) for coord in result['tour_coords']]
        try:
            i=a #TODO: Quitar
            if ('tour_coords' not in st.session_state) or (tour_coords != st.session_state['tour_coords']):
                coordinates = here.calculate_route_HERE(tour_coords)
                st.session_state['tour_coords'] = tour_coords
                st.session_state['coordinates'] = coordinates
            else: 
                coordinates = st.session_state['coordinates']
            folium.plugins.AntPath(locations=coordinates, dash_array=[8, 100], delay=800, 
                                   color='red').add_to(m)
        except:
            folium.plugins.AntPath(locations=result['tour_coords'], dash_array=[8, 100], 
                                   delay=800, color='red').add_to(m)
            st.warning("Rate limit for HERE service has been reached", icon="⚠️")
    st_data = st_folium(m, width=725)

    ###################################################################################################
    # ML PREDICTIONS            #######################################################################
    ###################################################################################################
    if 'nodes_predicted_demand' in result.keys():
        st.header('Prediction info for ' + str(result['method'])+ ':', divider='red')
        write_to_result_file(result_file_path, f'Method Objective function: {round(result['optimum_value'],2)}')


        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f' 📈 MSE: **{round(result["MSE"], 2)}**')
            write_to_result_file(result_file_path, f' MSE: {round(result["MSE"], 2)}')
        with col2:
            st.markdown(f' 📈 R2: **{round(result["R2"], 2)}**')
            write_to_result_file(result_file_path, f' R2: {round(result["R2"], 2)}')
        with col3:
            st.markdown(f' 📈 MAE: **{round(result["MAE"], 2)}**')
            write_to_result_file(result_file_path, f' MAE: {round(result["MAE"], 2)}')
        combined_data = []
        combined_data.append(('Real Demand',) + tuple(result['nodes_demand']))
        combined_data.append(('Predicted Demand',) + tuple(result['nodes_predicted_demand']))
        combined_data.append(('Train data size',) + tuple(result['n_train']))

        model_info = result['model_result'][0]['model_info']
        model_info_data = []
        for i in range(result['num_nodes']):
            model_info_node = result['model_result'][i][model_info]
            model_info_data.append(model_info_node)
        combined_data.append((model_info,) + tuple(model_info_data))

        df_ml = pd.DataFrame(combined_data, columns=['Variable'] + ['Node {}'.format(
            nodeDataSelected['codnode'].values[i]) for i in range(result['num_nodes'])])
        # Traspose DataFrame
        df_ml = df_ml.set_index('Variable').T
        df_ml.index = ['Node {}'.format(nodeDataSelected['codnode'].values[i]) 
                       for i in range(result['num_nodes'])]
        df_ml.to_csv(os.path.join(st.session_state["path_method"], 'info' + result['info'] +".csv"), sep=',', header=True)
        st.dataframe(df_ml, use_container_width=True)

    ###################################################################################################
    # MULTISCENARIO SIMULATIONS #######################################################################
    ###################################################################################################
    elif 'num_scenarios' in result.keys():
        st.header('Simulation info:', divider='red')
        d_ms = result['nodes_demand_multiscenario']
        params_simulation = result['params_simulations']
        combined_data = []
        mus = []
        sigmas = []
        demands = []
        for i in range(result['num_nodes']):
            mu, sigma = params_simulation[i]
            demand, bin = np.histogram(demandDataSelected[demandDataSelected['codnode'] == 
                                                          nodeDataSelected['codnode'].values[i]]['Pallets'].tolist())
            demands.append(demand)
            mus.append(mu)
            sigmas.append(sigma)
        combined_data.append(('Real Demand',) + tuple(result['nodes_demand']))
        combined_data.append(('μ',) + tuple(mus))
        combined_data.append(('σ',) + tuple(sigmas))
        combined_data.append(('distribution',) + tuple(demands))
        for s in range(result['num_scenarios']):
            combined_data.append(('Demand scenario {}'.format(s+1),) + tuple(d_ms[s]))
        
        df_ms = pd.DataFrame(combined_data, columns=['Variable'] + ['Node {}'.format(
            nodeDataSelected['codnode'].values[i]) for i in range(result['num_nodes'])])
        # Traspose DataFrame
        df_ms = df_ms.set_index('Variable').T
        df_ms.index = ['Node {}'.format(nodeDataSelected['codnode'].values[i]) 
                       for i in range(result['num_nodes'])]
        df_ms.to_csv(os.path.join(st.session_state["path_method"], 'info' + result['info'] +".csv"), sep=',', header=True)
        st.dataframe(df_ms, use_container_width=True, 
                     column_config={"distribution": 
                                    st.column_config.BarChartColumn("Demand Distribution", y_min=0, y_max=80),},)
    elif 'k' in  result.keys():
        st.header(f'Simulation info for {result['k']} Nearest Neighbour:', divider='red')
        d_ms = result['nodes_demand_multiscenario']
        min_max_dist = result['min_max_dist']
        combined_data = []
        demands = []
        range_distances = []
        for i in range(result['num_nodes']):
            min_dist, max_dist = min_max_dist[i]
            demand, bin = np.histogram(demandDataSelected[demandDataSelected['codnode'] == 
                                                          nodeDataSelected['codnode'].values[i]]['Pallets'].tolist())
            demands.append(demand)
            range_distances.append(str(round(min_dist,2)) + '-' + str(round(max_dist,2)))

        combined_data.append(('Real Demand',) + tuple(result['nodes_demand']))
        combined_data.append(('Range Distance',) + tuple(range_distances))
        combined_data.append(('distribution',) + tuple(demands))
        for i in range(result['k']):
            combined_data.append(('{}º neighbour demand'.format(i+1),) + tuple(d_ms[i]))
        
        df_ms = pd.DataFrame(combined_data, columns=['Variable'] + ['Node {}'.format(
            nodeDataSelected['codnode'].values[i]) for i in range(result['num_nodes'])])
        # Traspose DataFrame
        df_ms = df_ms.set_index('Variable').T
        df_ms.index = ['Node {}'.format(nodeDataSelected['codnode'].values[i]) 
                       for i in range(result['num_nodes'])]
        df_ms.to_csv(os.path.join(st.session_state["path_method"], 'info' + result['info'] +".csv"), sep=',', header=True)
        st.dataframe(df_ms, use_container_width=True, 
                column_config={"distribution": 
                            st.column_config.BarChartColumn("Demand Distribution", y_min=0, y_max=80),},)
        
        
    #################################################################################################
    # STADISTICS                #####################################################################
    #################################################################################################

    st.header('Stadistics:', divider='red')
    write_to_result_file(result_file_path, 'Stadistics:')

    st.markdown(f'🚚 Truck with a capacity of **{result['total_capacity']}** doing a distance of **{round(
        result['total_distance'],2)} Km**')
    write_to_result_file(result_file_path, f'Truck with a capacity of {result['total_capacity']} doing a distance of {round(
        result['total_distance'],2)} Km')
    st.markdown(f'**{result['num_visited']}** points have been visited using **{round(
        result['capacity_used'], 2)}** capacity unit')
    write_to_result_file(result_file_path, f'{result['num_visited']} points have been visited using {round(
        result['capacity_used'], 2)} capacity unit')

    # Compute percentages
    percentage_clients_delivered = (result['num_visited'] / result['num_nodes']) * 100
    write_to_result_file(result_file_path, f'Percentage Clients Delivered: {percentage_clients_delivered}%')
    percentage_delivered = (result['capacity_used'] / np.sum(result['nodes_demand'])) * 100
    write_to_result_file(result_file_path, f'Percentage Demand Delivered: {percentage_delivered}%')
    percentage_truck_filling = (result['capacity_used'] / result['total_capacity']) * 100
    write_to_result_file(result_file_path, f'Percentage Truck Filling: {percentage_truck_filling}%')

    # Show Compute percentages
    chart_data = pd.DataFrame(
        [percentage_clients_delivered, percentage_delivered, percentage_truck_filling],
        index = ["% Clients Delivered", "% Demand Delivered", "% Truck"]
    )
    # Apply data formatting for Altair
    data = pd.melt(chart_data.reset_index(), id_vars=["index"])

    # Assign colours depending on whether above or below 50 or 75
    def assign_color(value):
        if value < 50:
            return "#FFA07A"  # LightSalmon
        elif value < 75:
            return "#FFD700"  # Gold
        elif value <=100:
            return "#98FB98"  # PaleGreen
        else:
            return "#FFA07A"  # LightSalmon

    data['color'] = data['value'].apply(assign_color)

    # Stacked horizontal bar chart
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("value", type="quantitative", title=""),
            y=alt.Y("index", type="nominal", title=""),
            color=alt.Color("color:N", legend=None, scale=None),
        )
    )

    st.altair_chart(chart, use_container_width=True)

    ################################################################################################
    # DATA USED                #####################################################################
    ################################################################################################

    st.header('Data:', divider='red')
    # Show the data 
    st.subheader('Nodes info:')
    nodeDataSelected_df = dataframe_explorer(nodeDataSelected, case=False)
    st.dataframe(nodeDataSelected_df, use_container_width=True, hide_index = True)
    st.subheader('Demand info:')
    demandDataSelected_df = dataframe_explorer(demandDataSelected, case=False)
    st.dataframe(demandDataSelected_df, use_container_width=True, hide_index = True)

################################################################################################
# os.system('streamlit run c:/Users/ctff0/OneDrive/Escritorio/TFM/StocasticOpt/main.py')
