import networkx as nx
import matplotlib.pyplot as plt
import math


def is_number(s):
    try:
        float(s)  # or int(s) if you only want to check for integers
        return True
    except ValueError:
        return False
def create_map_from_file(filename="VPFS/coordinates.txt", navfile="VPFS/navigation coordinates", preferences="VPFS/edges.txt"):
    """
    Creates a NetworkX graph from a text file with coordinates, 
    connecting nodes with the same x or y coordinate.

    Args:
        filename (str): The name of the input file.

    Returns:
        nx.Graph: The graph representing the map.
    """

    G = nx.Graph()
    pos = {}  # Dictionary to store node positions
    nodes = {} # Dictionary to store node data (name and coordinates)

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(".")
            if len(parts) == 4:
                try:
                    name1 = parts[0]
                    name2 = parts[1]
                    name = name1 + name2
                    coordx = int(parts[2])  # Convert to integer
                    coordy = int(parts[3])  # Convert to integer
                    type = "stop"

                    G.add_node(name, type="stop")
                    pos[name] = (coordx, coordy)
                    nodes[name] = (name1, name2, coordx, coordy)

                except ValueError:
                    print(f"Skipping line due to invalid format: {line.strip()}")


    


    # Add edges based on coordinate adjacency (Corrected Logic)
    preferenceparts = {}

    with open(preferences) as f:
        for line in f:
            preferenceparts = line.strip().split(".")
            #print(preferenceparts)

    for node1, (_, _, x1, y1) in nodes.items():
        for node2, (_, _, x2, y2) in nodes.items():
            if node1 != node2:
                if x1 == x2 or y1 == y2:  # Connect if x OR y coordinates are the same
                    if is_number(node1)==False or is_number(node2)==False:
                        with open(preferences) as f:
                            for line in f:
                                #print(line)
                                preferenceparts = line.strip().split(".")
                                nodex, nodey, preference = preferenceparts

                                if (node1 == nodex) and (node2 == nodey):
                                    #print(nodex)
                                    #print(nodey)
                                        
                                    distance = math.sqrt(((abs(x1-x2))**2)+(abs(y1-y2))**2)
                                    #print(distance)
                                    #print(distance)
                                    #print(preference)
                                    G.add_edge(node1, node2, distance = distance, preference = preference)
                                    break



    with open(navfile, "r") as f:
        for line in f:
            parts = line.strip().split(".")
            if len(parts) == 3:  # Expecting node1, node2, preference
                try:
                    node1 = parts[0]  # Get the first node name (string)
                    node2 = parts [1] # Get the second node name (string)
                    preference = parts[2]  # Get the preference value

                    if node1 in pos and node2 in pos:  # Now using strings as keys
                        x1, y1 = pos[node1]
                        x2, y2 = pos[node2]
                        distance = math.sqrt(((abs(x1-x2))**2)+(abs(y1-y2))**2)
                        #print(distance)
                        G.add_edge(node1, node2, distance=distance, preference=preference)
                    else:
                        print(f"Skipping line in navfile due to missing nodes: {line.strip()}")

                except (ValueError, KeyError, IndexError) as e: # Catch multiple possible errors
                    print(f"Skipping line in navfile due to error: {e}, Line: {line.strip()}")

            else:
                print(f"Skipping line in navfile due to invalid number of parts: {line.strip()}")

    curvedEdgePoints = {

        

    }

    

    return G, pos

# Example usage (assuming your data is in "coordinates.txt"):
graph, positions = create_map_from_file("VPFS/coordinates.txt", preferences="VPFS/edges.txt")  # Or specify the filename

# Draw the graph
#nx.draw(graph, positions, with_labels=True, node_color="skyblue", node_size=300, edge_color="gray")
plt.show()


def save_edges_to_file(graph, filename="edges.txt"):
   
    try:
        with open(filename, "w") as f:
            for u, v in graph.edges():
                line = f"{u}.{v}.1\n"  # Create the formatted line
                f.write(line)  # Write to the file
        print(f"Edges saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving edges: {e}")

#save_edges_to_file(graph, "edges.txt")


#def findpath()