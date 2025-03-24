import json
from urllib import request
from maptest import create_map_from_file
import networkx as nx
import math

# Server details will change between lab, home, and competition, so saving them somehwere easy to edit
server_ip = "127.0.0.1"
server = f"http://{server_ip}:5000"
authKey = "7"
team = 7

graph, positions = create_map_from_file("VPFS/coordinates.txt")

def coordinatesToNode(x, y):
    with open("VPFS/coordinates.txt", 'r') as f:
        f.seek(0)
        for line in f:
            parts = line.strip().split(".")
            if ((parts[2] == str(x)) & (parts[3] == str(y))):
                return (parts[0]+parts[1])

def nodeToCoordinates(node):
    with open("VPFS/coordinates.txt", 'r') as f:
        f.seek(0)
        for line in f:
            parts = line.strip().split('.')
            if ((parts[0]+parts[1]) == node):
                x = parts[2]
                y = parts[3]
                #print(f"{x}, {y}")

                return x, y




            
                    


                

                    
        
        

#def estimateTripTime(distance, path, driveSpeed):
    #with open("coordinates.txt", 'r') as f:
        




            

##print(coordinatesToNode(446, 135))




# Make request to fares endpoint
def fareClaim():
    res = request.urlopen(server + "/fares")

    if res.status == 200:
    # Decode JSON data
        fares = json.loads(res.read())
    # Loop over the available fares
        for fare in fares:
            # If the fare is claimed, skip it
            if not fare['claimed']:
                
                print(fare)
    







  # Or specify the filename

def weighted_dijkstra(graph, source, target, weight='distance', preference='preference'):
    """Finds the shortest path considering both distance and preference."""

    def custom_weight(u, v, data):
        dist = data.get(weight, float('inf'))
        pref = data.get(preference, 0)

        try:
            dist = float(dist)
        except (ValueError, TypeError):
            dist = float('inf')

        try:
            pref = float(pref)
        except (ValueError, TypeError):
            pref = 0.0

        combined_weight = (distance_weight * dist) + (preference_weight * pref)
        return combined_weight

    distance_weight = 1.0  # Adjust this to change distance influence
    preference_weight = 0.5  # Adjust this to change preference influence

    try:
        path = nx.dijkstra_path(graph, source, target, weight=custom_weight)

        # Calculate the cost MANUALLY using custom_weight:
        cost = 0
        for i in range(len(path) - 1):  # Iterate over edges in the path
            u = path[i]
            v = path[i + 1]
            edge_data = graph.get_edge_data(u, v)

            if edge_data is not None:  # Check if the edge exists
                cost += custom_weight(u, v, edge_data)
            else:
                return None, None  # Handle cases where edges aren't found
        return cost, path

    except nx.NetworkXNoPath:
        return None, None


def find_shortest_path(graph, start_node, end_node):

    cost, path = weighted_dijkstra(graph, start_node, end_node)

    if path:
        return cost, path
    else:
        return None, None
    


#distance, path = find_shortest_path(graph, "77", "33")
#distaxx, sds = find_shortest_path(graph, "44", "77")

#sds.pop()

#path = sds + path

##print(distance)
##print(path)
##print(sds)

def estimateNode(x, y, tolerance):
    
    x = str(x)
    y = str(y)
    yLower = 0
    yUpper = 10000
    xLower = 0
    xUpper = 10000
    upair = 0
    lpair = 0
    
    with open("VPFS/coordinates.txt", 'r') as f:

        for line in f:
            ##print('x')
            parts = line.strip().split(".")
            if((abs(int(parts[2])-int(x))<tolerance) or (abs(int(parts[3])-int(y))<tolerance)):
                #print('y')
                if ((abs(int(parts[2])-int(x)))<(abs(int(parts[3])-int(y)))): #x is closer
                    xHold = parts[2]
                    f.seek(0)
                    for line in f:
                        if (parts[2]==xHold):
                            if ((int(parts[3]))>yLower and (int(parts[3])<int(y))):
                                yLower = int(parts[3])
                                lpair = int(parts[2])
                            elif ((int(parts[3]))<yUpper and (int(parts[3])>int(y))):
                                yUpper = int(parts[3])
                                upair = int(parts[2])

                    if (abs(yUpper-int(y))>abs(yLower-int(y))): #closer to upper bound
                        return coordinatesToNode(upair, yUpper)
                    else: return coordinatesToNode(lpair, yLower)


                if ((abs(int(parts[2])-int(x)))>(abs(int(parts[3])-int(y)))): #y is closer
                    yHold = parts[3]
                    f.seek(0)
                    for line in f:
                        if (parts[3]==yHold):
                            if ((int(parts[2]))>xLower and (int(parts[2])<int(y))):
                                xLower = int(parts[2])
                                lpair = int(parts[3])
                            elif ((int(parts[2]))<xUpper and (int(parts[2])>int(y))):
                                xUpper = int(parts[2])
                                upair = int(parts[3])
                                
                    if (abs(xUpper-int(x))>abs(xLower-int(x))): #closer to upper bound
                        return coordinatesToNode(upair, yUpper)
                    else: return (coordinatesToNode(lpair, yLower))
        return "get fucked"
    
def estimateLocation(filepath, target_x, target_y, direction):
    """
    Estimates the closest location from a file based on target coordinates and a direction.

    Args:
        filepath (str): Path to the file containing coordinates.
        target_x (float): Target x-coordinate.
        target_y (float): Target y-coordinate.
        direction (str): Direction to consider ('UP', 'DOWN', 'LEFT', 'RIGHT').

    Returns:
        str: Identifier of the closest coordinate in the specified direction, or None if none found.
    """

    closest_distance = float('inf')
    closest_coordinate = None

    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split('.')
            x = float(parts[-2])
            y = float(parts[-1])
            distance = math.sqrt((x - target_x)**2 + (y - target_y)**2)

            # Directional filtering
            if direction == 'UP' and y > target_y:
                valid_direction = True
            elif direction == 'DOWN' and y < target_y:
                valid_direction = True
            elif direction == 'LEFT' and x < target_x:
                valid_direction = True
            elif direction == 'RIGHT' and x > target_x:
                valid_direction = True
            else:
                valid_direction = False
                #print("get fucked")

            if valid_direction and distance < closest_distance:
                closest_distance = distance
                closest_coordinate = parts[0] + parts[1]

    return closest_coordinate





#testcoord = estimateLocation('VPFS/coordinates.txt', 22, 325)
#print(testcoord)


##print(estimateNode(145, 260, 40))





def tripValue(Xs, Ys, Xi, Yi, Xf, Yf, value):


    Xs = str(Xs)
    Ys = str(Ys)
    Xi = str(Xi)
    Yi = str(Yi)
    Xf = str(Xf)
    Yf = str(Yf)
    print(Xi)
    #print(Yi)
    if(Yi == '28'):
        Yi = '29'
    print(Yi)
    vehicleStartingNode = coordinatesToNode(Xs, Ys)
    startingNode = coordinatesToNode(Xi, Yi)
    
    print(startingNode)
    endingNode = coordinatesToNode(Xf, Yf)
    xxxxx, path = find_shortest_path(graph, startingNode, endingNode)
    xxxxy, path2 = find_shortest_path(graph, vehicleStartingNode, startingNode)
    path2.pop()
    path = path2+path
    straight = 1

    flag = 0
    directionflag = 0
    previousNode = [0]*3

    with open("VPFS/coordinates.txt", 'r') as f:

        netDriveValue = 0

        for node in path:
            ##print (node)
            if(flag == 0):
                previousNode[0] = Xi
                previousNode[1] = Yi
                previousNode[2] = startingNode
                flag = 1
                
                continue
            ##print(netDriveValue)

            f.seek(0)
            for line in f:
                ##print('1')
                parts = line.strip().split(".")
                if ((parts[0]+parts[1]) == node):
                    if(directionflag == 0): #initialize direction
                        if ((parts[2] != previousNode[0]) & (parts[3] == previousNode[1])): #X changed, y stayed the same
                            if (int(parts[2])>int(previousNode[0])): #moving right
                                previousDirection = 'RIGHT'
                            else: previousDirection = 'LEFT'

                        if ((parts[3] != previousNode[1]) & (parts[2] == previousNode[0])): #Y changed, x stayed the same
                            if (int(parts[3])>int(previousNode[1])): #moving up
                                previousDirection = 'UP'
                            else: previousDirection = 'DOWN'

                        directionflag = 1


                    if ((parts[2] != previousNode[0]) & (parts[3] == previousNode[1])): #X changed, y stayed the same
                        if (int(parts[2])>int(previousNode[0])): #moving right
                            if previousDirection == 'RIGHT':
                                straight = 1
                            else: straight = 0
                            previousDirection = 'RIGHT'
                        elif previousDirection == 'LEFT':
                            straight = 1
                            previousDirection = 'LEFT'
                        else: 
                            straight = 0
                            previousDirection = 'LEFT'



                    if ((parts[3] != previousNode[1]) & (parts[2] == previousNode[0])): #Y changed, x stayed the same
                        if (int(parts[3])>int(previousNode[1])): #moving up
                            if previousDirection == 'UP':
                                straight = 1
                            else: straight = 0
                            previousDirection = 'UP'
                        elif previousDirection == 'DOWN':
                            straight = 1
                            previousDirection = 'DOWN'
                        else: 
                            straight = 0
                            previousDirection = 'DOWN'


                    if ((parts[2] != previousNode[0]) & (parts[3] != previousNode[1])): #both x and y change
                        if (abs(int(parts[2])-int(previousNode[0]))>abs(int(parts[3])-int(previousNode[1]))): #greater x change
                            if (int(parts[2])>int(previousNode[0])): #positive x change
                                previousDirection = 'RIGHT'
                            else: previousDirection = 'LEFT'
                        elif (int(parts[3])>int(previousNode[1])): #positive y change
                            previousDirection = 'UP'
                        else: previousDirection = 'DOWN'

                    distance = math.sqrt(((abs(int(previousNode[0])-int(parts[2])))**2)+(abs(int(previousNode[1])-int(parts[3])))**2)
                    
                    if (straight == 0):
                        distance*=2
                    ##print (straight)
                    netDriveValue = netDriveValue + distance
                    ##print(netDriveValue)
                    break
                    
                    


        worth = value/netDriveValue
        return worth





testTrip = tripValue(123, 29, 446, 29, 446, 402, 100)


#print(testTrip)