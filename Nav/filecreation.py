def create_adjacent_edges(coordinate_file):
    intersections = {}
    with open(coordinate_file, 'r') as f:
        for line in f:
            parts = line.strip().split('.')
            if len(parts) == 4:
                street1 = parts[0]
                street2 = parts[1]
                x = parts[2]
                y = parts[3]
                intersection_name = street1 + street2
                intersections[intersection_name] = (int(x), int(y)) # Store as integers

    edges = set()
    intersection_names = list(intersections.keys())

    for i in range(len(intersection_names)):
        intersection1_name = intersection_names[i]
        x1, y1 = intersections[intersection1_name]

        for j in range(len(intersection_names)):
            if i == j:
                continue

            intersection2_name = intersection_names[j]
            x2, y2 = intersections[intersection2_name]

            # Check for adjacency along x-axis:
            if x1 == x2:
                # Find intersections with the same x but between y1 and y2
                between = False
                for k in range(len(intersection_names)):
                    if k == i or k == j:
                        continue
                    xk, yk = intersections[intersection_names[k]]
                    if xk == x1 and min(y1, y2) < yk < max(y1, y2):
                        between = True
                        break
                if not between:  # Only connect if no intersection in between
                    edge_str = ".".join(sorted([intersection1_name, intersection2_name])) + ".1"
                    edges.add(edge_str)

            # Check for adjacency along y-axis (similar logic):
            elif y1 == y2:
                between = False
                for k in range(len(intersection_names)):
                    if k == i or k == j:
                        continue
                    xk, yk = intersections[intersection_names[k]]
                    if yk == y2 and min(x1, x2) < xk < max(x1, x2):
                        between = True
                        break
                if not between:
                    edge_str = ".".join(sorted([intersection1_name, intersection2_name])) + ".1"
                    edges.add(edge_str)

    return edges

# Example usage (replace with your file path):
coordinate_file = "coordinates.txt"  # Or your actual file name
edges = create_adjacent_edges(coordinate_file)

with open("edges1.text", "w") as f:
    for edge in edges:
        line=edge+'\n'
        f.write(line)