import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from transport.database import SiouxFalls
from transport.location import GivenFlow, SingleFlow, MultiFlow
from transport.ue import LinkBased


data = SiouxFalls()
G = data.get_network()
ods = data.get_demands()
cost = {}
for n in G.nodes():
    cost[n] = 1
R = 15
ue_model = LinkBased(G, ods, R=R)
ue_model.opt()
z0 = np.array(list(ue_model.link_flow.values()))
links = ue_model.link_set
p = 6   # station limits for MultiFlow Model


def test_given_path_model():
    print("================================= Given Flow Model =================================")
    m = GivenFlow(G, ods, cost, R)
    print("All flows should be covered:")
    m.opt_all_cover()
    print("Routes:")
    routes = m.get_routes()
    for od in ods.keys():
        print(f"{od}: {routes[od]}.")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)
    print("Station cost should be limited:")
    m.opt_max_cover(limit_cost=3)
    print("Routes:")
    routes = m.get_routes()
    for od in ods.keys():
        print(f"{od}: {routes[od]}.")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)

def test_single_path_model():
    print("================================ Single Flow Model =================================")
    m = SingleFlow(G, ods, cost, R)
    m.opt()
    print("Routes:")
    routes = m.get_routes()
    for od in ods.keys():
        print(f"{od}: {routes[od]}.")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)

def test_multiple_flow_model(p, z0):
    print("=============================== Multiple Flow Model ================================")
    m = MultiFlow(G, ods, cost, R)
    print("Test shortest path assumption:")
    m.opt_shortest_path()
    print(f"detour: {m.get_detour_cost()}")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)
    print(f"Test shortest path and ue objective:")
    m.opt_sp_and_ue(p, z0)
    print(f"detour: {m.get_detour_cost()}")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)

def test_multiple_flow_model_by_links(p, z0, links):
    print("======================== Multiple Flow Model (given link) =========================")
    m = MultiFlow(G, ods, cost, R, links=links)
    print(f"Test ue objective model, p = {p}:")
    m.opt_ue(p, z0)
    print(f"detour: {m.get_detour_cost()}")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)
    print(f"Test so objective model, p = {p}:")
    m.opt_so(p, z0)
    print(f"detour: {m.get_detour_cost()}")
    stations = m.get_stations()
    data.show_highlight_nodes(stations)
    flows = m.get_link_flows()
    data.show_links_weight(flows)


test_given_path_model()
test_single_path_model()

# based on shortest paths
test_multiple_flow_model(p, z0)
# based on ue links
test_multiple_flow_model_by_links(p, z0, links)
