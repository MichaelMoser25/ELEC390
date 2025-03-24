

import json
from urllib import request
#from test import tripValue, estimateLocation, nodeToCoordinates, coordinatesToNode
import test
#import djiskra

# Server details will change between lab, home, and competition, so saving them somehwere easy to edit
server_ip = "127.0.0.1"
server = f"http://{server_ip}:5000"
authKey = "7"
team = 7

def whereAmI():
    x, y = test.nodeToCoordinates(test.estimateLocation('VPFS/coordinates.txt', 22, 325))
    return x, y

#print(whereAmI())

tripvalues = [0]
i = 0
def claimATrip():
# Make request to fares endpoint
    res = request.urlopen(server + "/fares")
    # Verify that we got HTTP OK
    if res.status == 200:
    # Decode JSON data
        fares = json.loads(res.read())
    # Loop over the available fares
        for fare in fares:
        # If the fare is claimed, skip it
            if not fare['claimed']:
                x = int(float(fare['src']['x'])*100)
                y = int(float(fare['src']['y'])*100)
                xDest = int(float(fare['dest']['x'])*100)
                yDest = int(float(fare['dest']['y'])*100)
                xinit, yinit = whereAmI()
                xinit = int(xinit)
                yinit = int(yinit)
                value = float(fare['pay'])
                tripvalues[i] = test.tripValue(xinit, yinit, x, y, xDest, yDest, value)
                print(tripvalues[i])

claimATrip()
        

