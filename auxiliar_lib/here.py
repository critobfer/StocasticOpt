import flexpolyline as fp
import requests
import streamlit as st
import os
from dotenv import load_dotenv

def calculate_route_HERE(coordinates: list[tuple[float, float]]):
 # Function that calls the HERE API using a list of nodes and a vehicle type.

    # parameters
    # coordinates -- Node list
    # vehicle -- Vehicle type car, truck

    # returns
    # The coordinates of the route.
    
    load_dotenv()
    hereAPIKey_1 = os.environ.get('API_KEY_1') 
    hereAPIKey_2 = os.environ.get('API_KEY_2')
    url = "https://router.hereapi.com/v8/routes?&transportMode=truck"
    origenPoint = coordinates[0]
    origin = '&origin=' + str(origenPoint[0]) + ',' + str(origenPoint[1])  # Origin
    destinationPoint = coordinates[len(coordinates) - 1]
    destination = '&destination=' + str(destinationPoint[0]) + ',' + str(destinationPoint[1])  # Final Point
    via = ''
    for pos in range(1, len(coordinates) - 1): # Path from the first until n-1
        point = coordinates[pos]
        via = via + '&via=' + str(point[0]) + ',' + str(point[1])
    urlQuery_1 = url + origin + destination + via + '!passThrough=true&return=polyline,summary' + '&apikey=' + hereAPIKey_1
    urlQuery_2 = url + origin + destination + via + '!passThrough=true&return=polyline,summary' + '&apikey=' + hereAPIKey_2
    try:
        dataRouteResponse = call_api(urlQuery_1)
    except:
        dataRouteResponse = call_api(urlQuery_2)
    coordsList = get_coordinates_list_from_here(dataRouteResponse)
    return coordsList

def call_api(url):
    st.success('Llamada a la API')
    # GET to the endpoint represented by the given url until it returns 200. It then returns a json representing the respons
    response = requests.get(url)
    if response.status_code != requests.codes.ok:  
        raise Exception
    else:
        data = response.json()
        return data


def get_coordinates_list_from_here(dataRouteResponse):
    # Processes the HERE polyline and passes it to coordinates.
    numberOfPolylines = len(dataRouteResponse['routes'][0]['sections'])
    coordsList = list()
    for sectionPos in range(numberOfPolylines):
        polyline = dataRouteResponse['routes'][0]['sections'][sectionPos]['polyline']
        coordsResponse = fp.decode(polyline)
        coordsList.extend(coordsResponse)
    return coordsList