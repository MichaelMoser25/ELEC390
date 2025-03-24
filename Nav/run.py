import json
from urllib import request
from maptest import create_map_from_file
import networkx as nx
import math
from test import coordinatesToNode, nodeToCoordinates, estimateLocation




def run(direction, x, y, speed):
    testcoord = estimateLocation('VPFS/coordinates.txt', x, y, direction)

    print(testcoord)


run("UP", 22, 330, 5)