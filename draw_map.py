import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import optimization_problem as op
# https://folium.streamlit.app/

# @st.cache_data #Te toma los argumentos cache_resource para machine learning 

#Configuración de la aplicación Streamlit
st.title('Visualisation of Objective Function Traveling Salesman Problem under Demand Uncertainty')
st.sidebar.title('Configuration')

# Seleccionar archivo desde el ordenador
nodeData = st.sidebar.file_uploader("Node File", type=["csv"])
if nodeData is not None:
    nodeData = pd.read_csv(nodeData, encoding='latin-1', sep=';')
    # st.write(nodeData)
demandData = st.sidebar.file_uploader("Demand File", type=["csv"])
if demandData is not None:
    demandData = pd.read_csv(demandData, encoding='latin-1', sep=';')
    # st.write(demandData)

# Controles desplegables y barras
method = st.sidebar.selectbox('Select Method', ['Deterministic', 'Multi-scenario', 'Machine Learning'])
num_nodes = st.sidebar.slider('Number of points', min_value=3, max_value=40, value=5)

# Visualización opcional de nodos
# Visualización opcional de nodos
lineType = 1
if st.sidebar.checkbox('Solve'):
    # Aquí puedes colocar el código que se ejecutará cuando se presione el botón
    # Calcular FO y ordern de los nodos dado el ,étodo y el num_nodos
    nodes, tour, objective_function_value = op.solve_problem(method=method, num_nodos=num_nodes, nodeData=nodeData, demandData=demandData) 
    num_visited = len(tour)
    st.write(f'{num_visited} points have been visited')

    # Mostrar la función objetivo
    st.write(f'Objective Function: {objective_function_value}')

    m = folium.Map(location=[nodes[0][0], nodes[0][1]], zoom_start=10)
    for i in range(num_nodes):
        # Agregar marcador
        folium.Marker( [nodes[i][0], nodes[i][1]],
            popup=str(i),
            tooltip="Liberty Bell" ).add_to(m)
        
    for i in range(num_visited):
        if i == num_visited - 1:
            folium.plugins.AntPath(
            locations=[[tour[i][0], tour[i][1]], [tour[0][0], tour[0][1]]],
            dash_array=[8, 100], delay=800, color='blue').add_to(m)
        else:         # Agregar trayecto
            folium.plugins.AntPath(
                locations=[[tour[i][0], tour[i][1]], [tour[i+1][0], tour[i+1][1]]],
                dash_array=[8, 100], delay=800, color='blue').add_to(m)
    st_data = st_folium(m, width=725)

####################################################################################
# streamlit run c:/Users/ctff0/OneDrive/Escritorio/TFM/StocasticOpt/draw_map.py
