# coordinates = calculateRouteHERE(coordinates)
import flexpolyline as fp
import requests

def calculate_route_HERE(coordinates):
    # Funcion que llama a la API de HERE usando una lista de nodos y un tipo de vehiculo.

    # Parametros
    # coordinates -- Lista de nodos
    # vehicle -- Tipo de vehiculo car, truck

    # Devuelve
    # Las coordenadas de la ruta junto a su distancia en KM y su tiempo en horas.
    
    hereAPIKey = 'ZmuIUyPPUobvj5-Vate_oFaug3AL7VbwFZX5KX3S9Yc'  # API utilizado para Here
    url = "https://router.hereapi.com/v8/routes?&transportMode=truck"
    origenPoint = coordinates[0]
    origin = '&origin=' + str(origenPoint[0]) + ',' + str(origenPoint[1])  # Punto origen
    destinationPoint = coordinates[len(coordinates) - 1]
    destination = '&destination=' + str(destinationPoint[0]) + ',' + str(destinationPoint[1])  # Punto Fin de Ruta
    via = ''
    for pos in range(1, len(coordinates) - 1): # Trayecto de nodos desde el segundo hasta el penultimo
        point = coordinates[pos]
        via = via + '&via=' + str(point[0]) + ',' + str(point[1])

    urlQuery = url + origin + destination + via + '!passThrough=true&return=polyline,summary' + '&apikey=' + hereAPIKey
    dataRouteResponse = call_api(urlQuery)

    routeInfo = list()
    coordsList = get_coordinates_list_from_here(dataRouteResponse)
    # routeDistance, routeTime = get_route_distance_time_here(dataRouteResponse)

    # formattedDistance = round(routeDistance / 1000, 2)  # FROM M TO KM
    # formattedTime = round((routeTime / 60) / 60, 2)  # FROM SECONDS TO HOURS
    routeInfo.append(coordsList)
    # routeInfo.append(formattedDistance)
    # routeInfo.append(formattedTime)
    return coordsList


def call_api(url):
    # Hace GET al endpoint representado por la url dado hasta que devuelva 200. Despues devuelve un json que representa la respuesta.
    response = requests.get(url)
    while response.status_code != requests.codes.ok:  # para respuesta de json distintas de 200
        response = requests.get(url)
    data = response.json()
    return data
        


def get_coordinates_list_from_here(dataRouteResponse):
    # Processa la polyline de HERE y la pasa a coordenadas.
    numberOfPolylines = len(dataRouteResponse['routes'][0]['sections'])
    coordsList = list()
    for sectionPos in range(numberOfPolylines):
        polyline = dataRouteResponse['routes'][0]['sections'][sectionPos]['polyline']
        coordsResponse = fp.decode(polyline)
        coordsList.extend(coordsResponse)
    return coordsList
    

def get_route_distance_time_here(dataRouteResponse):
    # Processa la distancia y tiempo de la ruta HERE.

    # Parametros
    # dataRouteResponse -- Objeto respuesta de la API de HERE que representa la ruta.

    # Devuelve
    # Distancia en metros y tiempo en segundos para una ruta dada.
    
    numberOfStops = len(dataRouteResponse['routes'][0]['sections'])
    routeDistance = 0
    routeTime = 0
    for sectionPos in range(numberOfStops):
        summaryInfo = dataRouteResponse['routes'][0]['sections'][sectionPos]['summary']
        routeDistance = routeDistance + summaryInfo['length']
        routeTime = routeTime + summaryInfo['duration']
    return routeDistance, routeTime