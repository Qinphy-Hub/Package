import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transport.database import SiouxFalls
from transport.location import GivenPath


data = SiouxFalls()
G = data.get_network()
ods = data.get_demands()
cost = {}
for n in G.nodes():
    cost[n] = 1
R = 15


def test_given_path_model():
    m = GivenPath(G, ods, cost, R)
    m.opt_all_cover()
    print("Routes:")
    routes = m.get_routes()
    for od in ods.keys():
        print(f"{od}: {routes[od]}.")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)

test_given_path_model()
