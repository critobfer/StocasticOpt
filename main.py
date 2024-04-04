import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
import folium # https://folium.streamlit.app/
from streamlit_folium import st_folium 
from streamlit_extras.dataframe_explorer import dataframe_explorer 
import here
import random
import deterministic as det
import multi_scenario as ms
import machine_learning as ml

def select_execution_data(demandData, nodeData, max_num_nodes, date):
    random.seed(10051347)
    codnode_list = list(demandData[demandData['Date'] == date]['codnode'].values)
    number_nodes = len(codnode_list)
    if number_nodes > max_num_nodes:
        codnode_list = random.sample(codnode_list, max_num_nodes)
    nodeDataModel = nodeData[nodeData['codnode'].isin(codnode_list)]
    demandDataModel = demandData[(demandData['Date'] < date) & (demandData['codnode'].isin(codnode_list))]
    realDemand= demandData[(demandData['Date'] == date) & (demandData['codnode'].isin(codnode_list))]
    return nodeDataModel, demandDataModel, realDemand

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

st.set_option('deprecation.showPyplotGlobalUse', False)

# @st.cache_data #Takes you cache_resource arguments for machine learning 

# Configuring the Streamlit application
st.title('Traveling Salesman Problem under Demand Uncertainty :truck:')
st.markdown('This application uses several methods to solve the stochastic problem.')
st.divider()

# SIDE BAR
st.sidebar.header('Configuration', divider='red')

# Select file from computer
nodeData = st.sidebar.file_uploader(":open_file_folder: Node File", type=["csv"])
if nodeData is not None:
    nodeData = pd.read_csv(nodeData, encoding='latin-1', sep=';')
    st.session_state["nodeData"] = nodeData
demandData = st.sidebar.file_uploader(":open_file_folder: Demand File", type=["csv"])
if demandData is not None:
    demandData = pd.read_csv(demandData, encoding='latin-1', sep=';')
    demandData['Date'] = pd.to_datetime(demandData['Date'])
    st.session_state["demandData"] = demandData

st.sidebar.subheader('Day we want to solve', divider='red')
if demandData is not None:
    year = st.sidebar.selectbox('Select the year', options=[2023, 2024])
    months = [month_dict[i] for i in list(demandData[demandData['Year'] == year]['Month'].sort_values())]
    month_name = st.sidebar.select_slider('Select the month', options=months)
    month = month_dict_reversed[month_name]
    days = list(demandData[(demandData['Year'] == year) & (demandData['Month'] == month)]['Day'].sort_values())
    day = st.sidebar.select_slider('Select the day', options=days)

    date = datetime(year, month, day)

st.sidebar.subheader('Method', divider='red')
# Drop-down controls and bars
method = st.sidebar.selectbox('Select Method', ['Deterministic', 'Multi-scenario', 'Machine Learning'])

# Others parameters for the methods
st.sidebar.subheader('Parameters', divider='red')
if method == 'Multi-scenario':
    num_scenarios = st.sidebar.number_input('Number of Scenarios', min_value=1, step=1, value=30)
    ms_option = st.sidebar.selectbox('Options', ['Maximum expectation', 'Conditional Value at Risk (CVaR)', 'Worst Case Analysis'])
elif method == 'Machine Learning':
    ml_option = st.sidebar.selectbox('Machine Learning Options', ['RNNs', 'CNNs', 'Attention Models', 'DNN', 'Ensemble Models', 'Gaussian Processes', 'Hidden Markov Models'])
max_num_nodes = st.sidebar.slider('Max number of points', min_value=3, max_value=40, value=15)

# Optional display of nodes
if st.sidebar.button('Solve', type='primary', use_container_width=True ):
    if "nodeData" not in st.session_state or "demandData" not in st.session_state:
        if "nodeData" not in st.session_state:
            st.warning('We need a node data file', icon="⚠️")
        if "demandData" not in st.session_state:
            st.warning('We need a demand data file', icon="⚠️")
        st.stop()

    nodeDataSelected, demandDataSelected, realDemand = select_execution_data(demandData, nodeData, max_num_nodes, date)

    # Mostrar el spinner
    if method == 'Deterministic':
        with st.spinner('Executing model'):
            result = det.execute(nodeData=nodeDataSelected, realDemand=realDemand, demandData=demandDataSelected) 
        st.empty()
        st.session_state["result"] = result
    elif method == 'Multi-scenario':
        if ms_option in ['Maximum expectation', 'Conditional Value at Risk (CVaR)']:
            with st.spinner('Executing model'):
                result = ms.execute(num_scenarios=num_scenarios, option=ms_option, nodeData=nodeDataSelected, demandData=demandDataSelected, realDemand=realDemand) 
            st.empty()
            st.session_state["result"] = result
        else:
            st.warning('We are working on it', icon="🔧")
            st.stop()
    elif method == 'Machine Learning':
        st.warning('We are working on it', icon="🔧")
        st.stop()
        with st.spinner('Executing model'):
            result = ml.execute(num_nodos=num_nodes, option=ml_option, nodeData=nodeData, demandData=demandData) 
        st.empty()
        st.session_state["result"] = result
    else:
        st.warning('We are working on it', icon="🔧")
        st.stop()

if "result" in st.session_state:
    result = st.session_state["result"]
    nodeDataSelected = result['nodeDataSelected']
    demandDataSelected = result['demandDataSelected']
    
    st.header('Result:', divider='red')

    # Show Objective Function
    st.subheader('**Objective Function:**')
    st.markdown(f'***Method Objective function:*** {round(result['optimum_value'],2)}')
    st.markdown(f'***Distance travelled + Undelivered demand:*** {round(result['total_distance'] + np.sum(result['nodes_demand']) - result['capacity_used'] ,2)}')

    m = folium.Map(location=[nodeDataSelected['latitude'].values.mean(), nodeDataSelected['longitude'].values.mean()], zoom_start=11, tiles = "CartoDB Positron")
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
            folium.plugins.AntPath(locations=coordinates, dash_array=[8, 100], delay=800, color='red').add_to(m)
        except:
            folium.plugins.AntPath(locations=result['tour_coords'], dash_array=[8, 100], delay=800, color='red').add_to(m)
            st.warning("Rate limit for HERE service has been reached", icon="⚠️")
    st_data = st_folium(m, width=725)

    if 'nodes_demand_multiscenario' in result.keys():
        st.header('Simulation info:', divider='red')
        d_ms = result['nodes_demand_multiscenario']
        params_simulation = result['params_simulations']
        combined_data = []
        mus = []
        sigmas = []
        demands = []
        for i in range(result['num_nodes']):
            mu, sigma = params_simulation[i]
            demand, bin = np.histogram(demandDataSelected[demandDataSelected['codnode'] == nodeDataSelected['codnode'].values[i]]['Pallets'].tolist())
            demands.append(demand)
            mus.append(mu)
            sigmas.append(sigma)
        combined_data.append(('μ',) + tuple(mus))
        combined_data.append(('σ',) + tuple(sigmas))
        combined_data.append(('distribution',) + tuple(demands))
        for s in range(result['num_scenarios']):
            combined_data.append(('Demand scenario {}'.format(s+1),) + tuple(d_ms[s]))
        
        df_ms = pd.DataFrame(combined_data, columns=['Variable'] + ['Node {}'.format(nodeDataSelected['codnode'].values[i]) for i in range(result['num_nodes'])])
        # Traspose DataFrame
        df_ms = df_ms.set_index('Variable').T
        df_ms.index = ['Node {}'.format(nodeDataSelected['codnode'].values[i]) for i in range(result['num_nodes'])]
        st.dataframe(df_ms, use_container_width=True, column_config={"distribution": st.column_config.BarChartColumn("Demand Distribution", y_min=0, y_max=80),},)

    st.header('Stadistics:', divider='red')

    st.markdown(f'🚚 Truck with a capacity of **{result['total_capacity']}** doing a distance of **{round(result['total_distance'],2)} Km**')
    st.markdown(f'**{result['num_visited']}** points have been visited using **{round(result['capacity_used'], 2)}** capacity unit')

    # Compute percentages
    percentage_clients_delivered = (result['num_visited'] / result['num_nodes']) * 100
    percentage_delivered = (result['capacity_used'] / np.sum(result['nodes_demand'])) * 100
    percentage_truck_filling = (result['capacity_used'] / result['total_capacity']) * 100
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
        else:
            return "#98FB98"  # PaleGreen

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

    # Show Compute percentages
    # st.markdown('**Percentage delivered:** ' + str(round(percentage_delivered,2)) + '%')
    # st.markdown('**Percentage of truck filling:** ' + str(round(percentage_truck_filling,2)) + '%')

    st.header('Data:', divider='red')
    # Show the data 
    st.subheader('Nodes info:')
    nodeDataSelected_df = dataframe_explorer(nodeDataSelected, case=False)
    st.dataframe(nodeDataSelected_df, use_container_width=True, hide_index = True)
    st.subheader('Demand info:')
    demandDataSelected_df = dataframe_explorer(demandDataSelected, case=False)
    st.dataframe(demandDataSelected_df, use_container_width=True, hide_index = True)

####################################################################################
# os.system('streamlit run c:/Users/ctff0/OneDrive/Escritorio/TFM/StocasticOpt/main.py')


