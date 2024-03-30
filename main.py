import streamlit as st
import pandas as pd
import folium # https://folium.streamlit.app/
from streamlit_folium import st_folium 
from streamlit_extras.dataframe_explorer import dataframe_explorer 
import here
import deterministic as det
import multi_scenario as ms
import machine_learning as ml
import here

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
    st.session_state["demandData"] = demandData

st.sidebar.subheader('Method', divider='red')
# Drop-down controls and bars
method = st.sidebar.selectbox('Select Method', ['Deterministic', 'Multi-scenario', 'Machine Learning'])

# Others parameters for the methods
st.sidebar.subheader('Parameters', divider='red')
if method == 'Multi-scenario':
    num_scenarios = st.sidebar.number_input('Number of Scenarios', min_value=1, step=1, value=30)
    ms_option = st.sidebar.selectbox('Options', ['Maximum expectation', 'Minimum variance', 'Minimum mean squared error', 'Risk aversion'])
elif method == 'Machine Learning':
    ml_option = st.sidebar.selectbox('Machine Learning Options', ['RNNs', 'CNNs', 'Attention Models', 'DNN', 'Ensemble Models', 'Gaussian Processes', 'Hidden Markov Models'])

num_nodes = st.sidebar.slider('Number of points', min_value=3, max_value=40, value=15)

# Optional display of nodes
if st.sidebar.button('Solve', type='primary', use_container_width=True ):
    if "nodeData" not in st.session_state or "demandData" not in st.session_state:
        if "nodeData" not in st.session_state:
            st.warning('We need a node data file', icon="⚠️")
        if "demandData" not in st.session_state:
            st.warning('We need a demand data file', icon="⚠️")
        st.stop()
    # Mostrar el spinner
    if method == 'Deterministic':
        with st.spinner('Executing model'):
            result = det.execute(num_nodos=num_nodes, nodeData=nodeData, demandData=demandData) 
        st.empty()
        st.session_state["result"] = result
    elif method == 'Multi-scenario':
        st.warning('We are working on it', icon="🔧")
        st.stop()
        with st.spinner('Executing model'):
            result = ms.execute(num_nodos=num_nodes, num_scenarios=num_scenarios, option=ms_option, nodeData=nodeData, demandData=demandData) 
    elif method == 'Machine Learning':
        st.warning('We are working on it', icon="🔧")
        st.stop()
        with st.spinner('Executing model'):
            result = ml.execute(num_nodos=num_nodes, option=ml_option, nodeData=nodeData, demandData=demandData) 
    else:
        st.warning('We are working on it', icon="🔧")
        st.stop()

if "result" in st.session_state:
    result = st.session_state["result"]
    nodeDataSelected = nodeData[nodeData['codnode'].isin(result['codnodes_selected'])]
    demandDataSelected = demandData[demandData['codnode'].isin(result['codnodes_selected'])]
    
    st.header('Result:', divider='red')

    # Show Objective Function
    st.subheader(f'**Objective Function:** {round(result['optimum_value'],2)}')

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
                <p style="display: inline-block; margin-left: 5px;"><strong>Coord:</strong> ({latitude}, {longitude})</p>
            </div>
        </div>
        """.format(demand=result['nodes_demand'][i],
                    latitude=round(nodeDataSelected['latitude'].values[i], 3),
                    longitude=round(nodeDataSelected['longitude'].values[i], 3))
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

    st.header('Stadistics:', divider='red')

    st.markdown(f'{result['num_visited']} points have been visited using {round(result['capacity_used'], 2)} capacity unit')

    # Compute percentages
    percentage_delivered = (result['num_visited'] / result['num_nodes']) * 100
    percentage_truck_filling = (result['capacity_used'] / result['total_capacity']) * 100

    # Show Compute percentages
    st.markdown('**Percentage delivered:** ' + str(round(percentage_delivered,2)) + '%')
    st.markdown('**Percentage of truck filling:** ' + str(round(percentage_truck_filling,2)) + '%')

    st.header('Data:', divider='red')
    # Show the data 
    st.subheader('Nodes info:')
    nodeDataSelected_df = dataframe_explorer(nodeDataSelected, case=False)
    st.dataframe(nodeDataSelected_df, use_container_width=True)
    st.subheader('Demand info:')
    demandDataSelected_df = dataframe_explorer(demandDataSelected, case=False)
    st.dataframe(demandDataSelected_df, use_container_width=True)

####################################################################################
# os.system('streamlit run c:/Users/ctff0/OneDrive/Escritorio/TFM/StocasticOpt/main.py')
