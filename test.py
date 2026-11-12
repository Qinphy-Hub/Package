import gurobipy as gp
from gurobipy import GRB
import networkx as nx

G = nx.DiGraph()

# 节点（只有 1、3 有负荷）
G.add_node(0, load=(0.0, 0.0))   # 根节点
G.add_node(1, load=(0.5, 0.2))
G.add_node(2)                    # 无负荷
G.add_node(3, load=(0.6, 0.2))

# 支路
G.add_edge(0, 1, r=0.01, x=0.05, s_max=2.0)
G.add_edge(1, 2, r=0.01, x=0.05, s_max=2.0)
G.add_edge(2, 3, r=0.01, x=0.05, s_max=2.0)

# =========================
# 模型
# =========================
m = gp.Model("Baran_Wu_SOCP")

# =========================
# 变量
# =========================
P = m.addVars(G.edges(), lb=-GRB.INFINITY, name="P")
Q = m.addVars(G.edges(), lb=-GRB.INFINITY, name="Q")
v = m.addVars(G.nodes(), lb=0.95**2, ub=1.05**2, name="v")
l = m.addVars(G.edges(), lb=0.0, name="l")

# =========================
# 根节点（平衡节点）
# =========================
m.addConstr(v[0] == 1.0, "slack_voltage")

# =========================
# 1. 功率平衡
# =========================
for i in G.nodes():
    if i == 0:
        continue

    inflow_P = gp.quicksum(P[j, i] for j in G.predecessors(i))
    outflow_P = gp.quicksum(P[i, j] for j in G.successors(i))
    load_p, _ = G.nodes[i].get("load", (0.0, 0.0))

    m.addConstr(inflow_P - outflow_P == load_p, name=f"P_balance_{i}")

    inflow_Q = gp.quicksum(Q[j, i] for j in G.predecessors(i))
    outflow_Q = gp.quicksum(Q[i, j] for j in G.successors(i))
    _, load_q = G.nodes[i].get("load", (0.0, 0.0))

    m.addConstr(inflow_Q - outflow_Q == load_q, name=f"Q_balance_{i}")

# =========================
# 2. 电压 drop（精确 DistFlow）
# =========================
for i, j in G.edges():
    r = G.edges[i, j]["r"]
    x = G.edges[i, j]["x"]
    z2 = r**2 + x**2

    m.addConstr(
        v[i] - v[j] == 2 * (r * P[i, j] + x * Q[i, j]) - z2 * l[i, j],
        name=f"vdrop_{i}_{j}"
    )

# =========================
# 3. 二阶锥约束（核心）
# =========================
for i, j in G.edges():
    # P_{ij}^2 + Q_{ij}^2 <= v_i * l_{ij}
    m.addConstr(
        P[i, j] * P[i, j] + Q[i, j] * Q[i, j] <= v[i] * l[i, j],
        name=f"soc_{i}_{j}"
    )

# =========================
# 目标：最小化网损
# =========================
loss = gp.quicksum(
    G.edges[i, j]["r"] * l[i, j]
    for i, j in G.edges()
)
m.setObjective(loss, GRB.MINIMIZE)

# =========================
# 求解
# =========================
m.Params.QCPDual = 1      # 输出对偶（可选）
m.optimize()

# =========================
# 结果输出
# =========================
if m.status == GRB.OPTIMAL:
    print("\n=== Baran & Wu SOCP 最优结果 ===")
    print(f"总网损: {m.objVal:.4f} p.u.")
    print("\n--- 支路 ---")
    for i, j in G.edges():
        print(f"Branch {i}->{j}: "
              f"P={P[i,j].X:>7.3f}, Q={Q[i,j].X:>7.3f}, "
              f"l={l[i,j].X:.4f}, "
              f"|S|={((P[i,j].X)**2+(Q[i,j].X)**2)**0.5:.3f}")
    print("\n--- 节点电压 ---")
    for i in G.nodes():
        print(f"Node {i}: V={v[i].X**0.5:.4f} p.u.")
else:
    print("未找到最优解, status:", m.status)