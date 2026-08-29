import sys
from pathlib import Path
import math

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transport.database import SiouxFalls
from transport.ue import PathBased, LinkBased


data = SiouxFalls()
G = data.get_network()
ods = data.get_demands()


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

# TEST ONE
def test1():
    print("========================== TEST ONE: Link-based BPR(DEFAULT) of UE and SO ==========================")
    # basic function test (UE)
    ue_model = LinkBased(G, ods)
    ue_z = ue_model.opt()
    ue_t = ue_model.get_system_time_cost()
    print("The value of UE objective function is", ue_z)
    print("UE: the time cost of the system is", ue_t)
    # SO basic function test
    so_model = LinkBased(G, ods, ue_type='SO')
    so_z = so_model.opt()
    so_t = so_model.get_system_time_cost()
    print("The value of SO objective function is", so_z)
    print("SO: the time cost of the system is", so_t)
    # PoA
    print("The PoA of this system is:", ue_t / so_t)

# TEST TWO
def test2():
    print("============================= TEST TWO: Link-based UE and SO (CONICAL) =============================")
    # UE model
    ue_model = LinkBased(G, ods, LPF=conical, INT_LPF=INT_conical)
    ue_z = ue_model.opt()
    ue_t = ue_model.get_system_time_cost()
    print("The value of UE objective function is", ue_z)
    print("UE: the time cost of the system is", ue_t)
    # SO model
    so_model = LinkBased(G, ods, ue_type='SO', LPF=conical, INT_LPF=INT_conical, DER_LPF=DER_conical)
    so_z = so_model.opt()
    so_t = so_model.get_system_time_cost()
    print("The value of SO objective function is:", so_z)
    print("the time cost of the system is:", so_t)
    # PoA
    print("The PoA of this system is:", ue_t / so_t)

# TEST THREE
def test3():
    print("====================== TEST THREE: Path-based of UE and SO (BPR, None Paths) =======================")
    # basic function test (UE)
    ue_model = PathBased(G, ods, None)
    ue_z = ue_model.opt()
    ue_t = ue_model.get_system_time_cost()
    print("The value of UE objective function is", ue_z)
    print("UE: the time cost of the system is", ue_t)
    # # SO basic function test
    # so_model = PathBased(G, ods, None, ue_type='SO')
    # so_z = so_model.opt()
    # so_t = so_model.get_system_time_cost()
    # print("The value of SO objective function is", so_z)
    # print("SO: the time cost of the system is", so_t)
    # # PoA
    # print("The PoA of this system is:", ue_t / so_t)
    return ue_model.Paths

# TEST FOUR
def test4(Paths):
    print("======================== TEST FOUR: Path-based UE and SO (BPR, Given Paths) ========================")
    # basic function test (UE)
    ue_model = PathBased(G, ods, Paths)
    ue_z = ue_model.opt()
    ue_t = ue_model.get_system_time_cost()
    print("The value of UE objective function is", ue_z)
    print("UE: the time cost of the system is", ue_t)
    # SO basic function test
    so_model = PathBased(G, ods, Paths, ue_type='SO')
    so_z = so_model.opt()
    so_t = so_model.get_system_time_cost()
    print("The value of SO objective function is", so_z)
    print("SO: the time cost of the system is", so_t)
    # PoA
    print("The PoA of this system is:", ue_t / so_t)

# TEST FIVE
def test5(Paths):
    print("========================================== TEST FIVE: SUE ==========================================")
    sue_model = PathBased(G, ods, Paths, ue_type='SUE')
    sue_z = sue_model.opt()
    sue_t = sue_model.get_system_time_cost()
    print("The value of SUE objective function is:", sue_z)
    print("the time cost of the system is:", sue_t)


test1()
test2()
Paths = test3()
test4(Paths)
test5(Paths)


""" Results
========================== TEST ONE: Link-based BPR(DEFAULT) of UE and SO ==========================
The value of UE objective function is 4231359.049374299
UE: the time cost of the system is 7480197.278500641
The value of SO objective function is 7194407.8178115785
SO: the time cost of the system is 7194407.8178115785
The PoA of this system is: 1.0397238338340395
============================= TEST TWO: Link-based UE and SO (CONICAL) =============================
The value of UE objective function is 5610983.964860248
UE: the time cost of the system is 8616513.368459614
The value of SO objective function is: 7118133.143786397
the time cost of the system is: 7118133.143786397
The PoA of this system is: 1.210501854124658
====================== TEST THREE: Path-based of UE and SO (BPR, None Paths) =======================
The value of UE objective function is 4231359.049374299
UE: the time cost of the system is 7480197.278500641
The value of SO objective function is 7194407.8178115785
SO: the time cost of the system is 7194407.8178115785
The PoA of this system is: 1.0397238338340395
======================== TEST FOUR: Path-based UE and SO (BPR, Given Paths) ========================
The value of UE objective function is 4231359.049374299
UE: the time cost of the system is 7480197.278500641
The value of SO objective function is 7194457.442243401
SO: the time cost of the system is 7194457.442243401
The PoA of this system is: 1.039716662243281
========================================== TEST FIVE: SUE ==========================================
The value of SUE objective function is: 5112495.472745353
the time cost of the system is: 7418003.328157724
"""