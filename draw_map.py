import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
# https://folium.streamlit.app/

nodos_madrid = np.array([
    [40.4168, -3.7038],   # Madrid
    [40.4254, -3.6903],   # Puerta del Sol
    [40.4144, -3.7074],   # Plaza Mayor
    [40.4155, -3.7073],   # Palacio Real
    [40.4152, -3.7130],   # Almudena Cathedral
    [40.4054, -3.6887],   # Retiro Park
    [40.4300, -3.6922],   # Gran Vía
    [40.4194, -3.6918],   # Prado Museum
    [40.4239, -3.6924],   # Reina Sofía Museum
    [40.4232, -3.6925]    # Thyssen-Bornemisza Museum
])

@st.cache_data #Te toma los argumentos cache_resource para machine learning 
def calcular_funcion_objetivo(metodo, num_nodos):
    # Aquí puedes implementar tu lógica para calcular la función objetivo
    # en función del método seleccionado y el número de nodos
    # En este ejemplo, solo se genera una función aleatoria
    
    # Generar coordenadas aleatorias para los nodos
    np.random.seed(0)
    nodos = np.random.rand(num_nodos, 2)
    
    # Calcular la distancia euclidiana entre nodos
    distancias = np.linalg.norm(nodos[:, np.newaxis, :] - nodos[np.newaxis, :, :], axis=-1)
    
    # Calcular la función objetivo
    funcion_objetivo = np.sum(distancias)
    
    return funcion_objetivo

#Configuración de la aplicación Streamlit
st.title('Visualización de Función Objetivo TSP')
st.sidebar.title('Configuración')

# Controles desplegables y barras
metodo = st.sidebar.selectbox('Select Method', ['Deterministic', 'Multi-scenario', 'Machine Learning'])
num_nodos = st.sidebar.slider('Number of points', min_value=3, max_value=10, value=5)

# Calcular la función objetivo
funcion_objetivo = calcular_funcion_objetivo(metodo, num_nodos)


# Mostrar la función objetivo
st.write(f'Objective Function: {funcion_objetivo}')

# Visualización opcional de nodos
# Visualización opcional de nodos
lineType = 1
if st.checkbox('Show points'):
    m = folium.Map(location=[nodos_madrid[0][0], nodos_madrid[0][1]], zoom_start=16)
    for i in range(num_nodos):
        # Agregar marcador
        folium.Marker( [nodos_madrid[i][0], nodos_madrid[i][1]],
            popup=str(i),
            tooltip="Liberty Bell" ).add_to(m)
        
        if i == num_nodos - 1:
            folium.plugins.AntPath(
            locations=[[nodos_madrid[i][0], nodos_madrid[i][1]], [nodos_madrid[0][0], nodos_madrid[0][1]]],
            dash_array=[8, 100], delay=800, color='blue').add_to(m)
        else:         # Agregar trayecto
            folium.plugins.AntPath(
                locations=[[nodos_madrid[i][0], nodos_madrid[i][1]], [nodos_madrid[i+1][0], nodos_madrid[i+1][1]]],
                dash_array=[8, 100], delay=800, color='blue').add_to(m)
    st_data = st_folium(m, width=725)
else:
    # Convertir a DataFrame de pandas
    df = pd.DataFrame(nodos_madrid, columns=['LAT', 'LON'])

    # Mostrar el mapa
    st.map(data=df)

####################################################################################
# streamlit run c:/Users/ctff0/OneDrive/Escritorio/TFM/StocasticOpt/draw_map.py