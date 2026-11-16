from powergrid import database, macro


data = database.Norwegian()
m = macro.PowerFlow(data.topology, 1)
m.set_gen_and_cost({1: 5.0})
m.opt_conic_relax(data.get_load_data())
# m.opt_linear_approx(data.get_load_data())
# data.example_power_flow()

# data = database.IEEE33()
# m = powerflow.PowerFlow(data.grid, data.slack)
# m.set_gen_and_cost(data.cost)
# m.opt_conic_relax(data.load)
# # m.opt_linear_approx(data.load)