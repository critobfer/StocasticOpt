import streamlit as st
import numpy as np
import pandas as pd
import folium # https://folium.streamlit.app/
from streamlit_folium import st_folium 
import optimization_problem as op
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
demandData = st.sidebar.file_uploader(":open_file_folder: Demand File", type=["csv"])
if demandData is not None:
    demandData = pd.read_csv(demandData, encoding='latin-1', sep=';')
st.sidebar.divider()

# Drop-down controls and bars
method = st.sidebar.selectbox('Select Method', ['Deterministic', 'Multi-scenario', 'Machine Learning'])
num_nodes = st.sidebar.slider('Number of points', min_value=3, max_value=40, value=5)

# Optional display of nodes
if st.sidebar.button('Solve', type='primary', use_container_width=True ):
    if nodeData is None or demandData is None:
        st.warning('Please upload both node data and demand data files.')
    else:
        with st.spinner('Executing model...'):
            result = op.solve_problem(method=method, num_nodos=num_nodes, nodeData=nodeData, demandData=demandData)
            st.session_state["result"] = result
        st.success('Model executed successfully!')

    st.empty()
    st.session_state["result"] = result
    

if "result" in st.session_state:
    result = st.session_state["result"]

    nodeDataSelected = nodeData[nodeData['codnode'].isin(result['codnodes_selected'])]

    st.write(f'{result['num_visited']} points have been visited')

    # Show Objective Function
    st.write(f'Objective Function: {round(result['optimum_value'],2)}')

    m = folium.Map(location=[nodeDataSelected['latitude'].values[0], nodeDataSelected['longitude'].values[0]], zoom_start=10, tiles = "CartoDB Positron")
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
    coordinates = here.calculateRouteHERE(result['tour_coords'])
    folium.plugins.AntPath(locations=coordinates, dash_array=[8, 100], delay=800, color='red').add_to(m)

    st_data = st_folium(m, width=725)

    st.stop()

####################################################################################
# os.system('streamlit run c:/Users/ctff0/OneDrive/Escritorio/TFM/StocasticOpt/main.py')
