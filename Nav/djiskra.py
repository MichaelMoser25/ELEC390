from maptest import create_map_from_file
import networkx as nx



graph, positions = create_map_from_file("coordinates.txt")  # Or specify the filename

def weighted_dijkstra(graph, source, target, weight='distance', preference='preference'):
    """Finds the shortest path considering both distance and preference."""

    def custom_weight(u, v, data):
        dist = data.get(weight, float('inf'))
        pref = data.get(preference, 0)
        #print(dist)
        try:
            dist = float(dist)
        except (ValueError, TypeError):
            print(f"Warning: Distance '{dist}' for edge ({u}, {v}) is not a valid number. Using default.")
            dist = float('inf')

        try:
            pref = float(pref)
        except (ValueError, TypeError):
            print(f"Warning: Preference '{pref}' for edge ({u}, {v}) is not a valid number. Using default.")
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
            v = path[i+1]
            edge_data = graph.get_edge_data(u, v)
          
            if edge_data is not None: # Check if the edge exists
                cost += custom_weight(u, v, edge_data)
            else:
                return None, None # Handle cases where edges aren't found
        return cost, path

    except nx.NetworkXNoPath:
        return None, None


def find_shortest_path(graph, start_node, end_node):

    cost, path = weighted_dijkstra(graph, start_node, end_node)

    if path:
        return cost, path
    else:
        return None, None
    


distance, path = find_shortest_path(graph, "77", "33")  


print(distance)
print(path)

