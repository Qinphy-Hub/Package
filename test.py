from powergrid import database, macro
import pandas as pd
import math

# data = database.Norwegian()
# data.run_powerflow(10)

data = database.Norwegian()
m = macro.BranchFlow(data.topology, data.slack)
load = {}
for col in data.load_data.columns:
    p = data.load_data[col][10]
    q = math.sqrt((p / data.power_factor) ** 2 - p ** 2)
    load[int(col)] = (p / 10, q / 10)

m.set_gen_and_cost({1: 5.0})
m.opt_conic_relax(load)
