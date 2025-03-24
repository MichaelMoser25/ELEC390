import networkx as nx
import matplotlib.pyplot as plt

def create_map_from_file(filename="coordinates.txt"):
    """
    Creates a NetworkX graph from a text file with coordinates.

    Args:
        filename (str): The name of the input file.

    Returns:
        nx.Graph: The graph representing the map.
    """

    G = nx.Graph()
    pos = {}  # Dictionary to store node positions

    with open(filename, "r") as f:
        previous = None
        edges = []
        for line in f:
            parts = line.strip().split(".") 
            print(parts)
            
            if len(parts) == 4:  
                try:
                   
                    name1 = parts[0]
                    name2 = parts[1]
                    name = name1+"-"+name2
                    coordx = parts[2]
                    coordy = parts[3]
                    coordx=int(coordx)
                    coordy=int(coordy)
                    # Add nodes with positions
                    G.add_node(name)
                    pos[name] = (coordx, coordy)
                    #print(int(coordx))
                    #print(int(coordy))

                    if previous:
                        edges.append((previous, name))
                    previous = name

                    # Add edge
                    #G.add_edge(name1, name2)

                except ValueError:
                    print(f"Skipping line due to invalid format: {line.strip()}")

        G.add_edges_from(edges)
        
    return G, pos

# Create the graph and get node positions
graph, positions = create_map_from_file()

# Draw the graph
nx.draw(graph, positions, with_labels=True, node_color="skyblue", node_size=300, edge_color="gray")
plt.show()





