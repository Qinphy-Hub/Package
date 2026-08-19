import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transport.database import SiouxFalls
from transport.ue import ClassicalFW
import math

data = SiouxFalls(multi_dis=1)
G = data.get_network()
ods = data.get_demands()

# TEST ONE
print("TEST ONE: BPR(DEFAULT) ==================")

# basic function test (UE)
ue_model = ClassicalFW(G, ods)
ue_z = ue_model.ue_opt()
ue_t = ue_model.get_time_cost()
print("The value of UE objective function is:", ue_z)
print("the time cost of the system is:", ue_t)

# SO basic function test
so_model = ClassicalFW(G, ods, type='SO')
so_z = so_model.so_opt()
so_t = so_model.get_time_cost()
print("The value of SO objective function is:", so_z)
print("the time cost of the system is:", so_t)

# PoA
print("The PoA of this system is:", ue_t / so_t)

# TEST TWO
print("TEST TWO: CONICAL(CUSTOM) ==================")

# custom link performance function
def conical(FFT, C, flow, alpha=0.25, beta=1):
    return FFT * (2 + math.sqrt((alpha ** 2) * ((1 - flow / C) ** 2) + (beta ** 2)) - alpha * (1 - flow / C) - beta)

def INT_conical(FFT, C, flow, alpha=0.25, beta=1):
    u = flow / C
    a = C * FFT * ((2 - beta - alpha) * u + alpha / 2 * (u ** 2))
    b = C * FFT / 2 * ((1 - u) * math.sqrt((alpha ** 2) * ((1 - u) ** 2) + (beta ** 2)) - math.sqrt((alpha ** 2) + (beta ** 2)))
    c = C * FFT * (beta ** 2) / (2 * alpha) * math.log10((math.sqrt((alpha ** 2) * ((1 - u) ** 2) + (beta ** 2)) + alpha * (1 - u)) / (math.sqrt((alpha ** 2) + (beta ** 2)) + alpha))
    return a - b - c

def DER_conical(FFT, C, flow, alpha=0.25, beta=1):
    u = flow / C
    a = (alpha ** 2) * (1 - u)
    b = math.sqrt((alpha ** 2) * ((1 - u) ** 2) + (beta ** 2))
    return FFT / C * (alpha - a / b)

cc_model = ClassicalFW(G, ods, LPF=conical, INT_LPF=INT_conical)
cc_z = cc_model.ue_opt()
cc_t = cc_model.get_time_cost()
print("The value of UE objective function is:", cc_z)
print("the time cost of the system is:", cc_t)

co_model = ClassicalFW(G, ods, type='SO', LPF=conical, INT_LPF=INT_conical, DER_LPF=DER_conical)
co_z = co_model.so_opt()
co_t = co_model.get_time_cost()
print("The value of SO objective function is:", co_z)
print("the time cost of the system is:", co_t)

# PoA
print("The PoA of this system is:", cc_t / co_t)

# TEST THREE
# print("TEST THREE: SUE ==================")
