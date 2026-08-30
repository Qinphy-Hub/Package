from simple import Model, SourceMMc
import math


def test_MM1():
    m = Model(c=1)
    m.set_data(rate=5, mu=6)
    print("================================================== TEST M/M/1 ==================================================")
    print(f"Traffic intensity = {m.get_traffic_intensity()}, Answer is {5.0 / 6.0}")
    print(f"System length = {m.get_system_length()}, Answer is 5")
    print(f"Queue length = {m.get_queue_length()}, Answer is {4 + 1.0 / 6.0}")
    print(f"P0 = {m.get_P_n(0)}, Answer is {1.0 / 6.0}")
    print(f"{m.get_probability_of_queue_wait_time(0)} get service immediatly, Answer is 16.7%")
    print(f"System wait time = {m.get_system_wait_time()}, Answer is 1")
    print(f"Queue wait time = {m.get_queue_wait_time()}, Answer is {5.0 / 6.0}")
    print(f"The probability of queuing time more than 45 is {1 - m.get_probability_of_queue_wait_time(45.0 / 60.0)}, Answer is {5.0 / 6.0 * math.exp(-3.0 / 4.0)}")

def test_MMc():
    print("================================================== TEST M/M/c ==================================================")
    print("Model: c=3")
    m = Model(c=3)
    print("lambda=30, mu=12:")
    m.set_data(rate=30, mu=12)
    print(f"Offered load is {m.get_offered_load()}, Answer is 2.5")
    print(f"Traffic intensity = {m.get_traffic_intensity()}, Answer is {5.0 / 6.0}")
    print(f"The value of Erlang C formula is {m.ErlangC()}, Answer is 0.702")
    print("lambda=6, mu=3:")
    m.set_data(rate=6, mu=3)
    print(f"Offered load is {m.get_offered_load()}, Answer is 2")
    print(f"Traffic intensity = {m.get_traffic_intensity()}, Answer is {2.0 / 3.0}")
    print(f"P0 = {m.get_P_n(0)}, Answer is {1.0 / 9.0}")
    print(f"Queue length = {m.get_queue_length()}, Answer is {8.0 / 9.0}")
    print(f"System wait time = {m.get_system_wait_time()}, Answer is {13.0 / 27.0}")
    print("Model: c=4")
    m = Model(c=4)
    m.set_data(rate=6, mu=3)
    print(f"The probability of service imediatly is {1 - m.get_probability_of_queue_wait_time(0)}, Answer is {4.0 / 23.0}")
    print(f"Queue length = {m.get_queue_length()}, Answer is {4.0 / 23.0}")
    print(f"The value of Erlang C (4, 2) is {m.ErlangC(4, 2)}, Answer is {4.0 / 23.0}")

def test_MMcK():
    m = Model(c=3, K=7)
    print("================================================= TEST M/M/c/K =================================================")
    print(f"lambda=1, mu={1.0 / 6.0}:")
    m.set_data(rate=1, mu=1.0/6.0)
    print(f"Offered load is {m.get_offered_load()}, Answer is 6")
    print(f"Traffic intensity = {m.get_traffic_intensity()}, Answer is 2")
    print(f"P0 = {m.get_P_n(0)}, Answer is {1.0 / 1141.0}")
    print(f"Queue length = {m.get_queue_length()}, Answer is 3.09")
    print(f"System length = {m.get_system_length()}, Answer is 6.06")
    print(f"System wait time = {m.get_system_wait_time()}, Answer is 12.3")

def test_MMcc():
    print("================================================= TEST M/M/c/c =================================================")
    m = Model(c=4, K=4)
    m.set_data(rate=6, mu=3)
    print(f"Offered load is {m.get_offered_load()}, Answer is 2")
    print(f"Traffic intensity = {m.get_traffic_intensity()}, Answer is 0.5")
    print(f"The value of Erlang B (0, 2) is {m.ErlangB(0, 2)}, Answer is 1")
    print(f"The value of Erlang B (1, 2) is {m.ErlangB(1, 2)}, Answer is {2.0 / 3.0}")
    print(f"The value of Erlang B (2, 2) is {m.ErlangB(2, 2)}, Answer is {2.0 / 5.0}")
    print(f"The value of Erlang B (3, 2) is {m.ErlangB(3, 2)}, Answer is {4.0 / 19.0}")
    print(f"The value of Erlang B (4, 2) is {m.ErlangB(4, 2)}, Answer is {2.0 / 21.0}")

def test_infinite_capacity():
    print("================================================= TEST M/M/inf =================================================")
    m = Model()
    m.set_data(rate=20000, mu=1.0/1.5)
    print(f"System length = {m.get_system_length()}, Answer is 30000")

def test_sourceMMc():
    print("============================================== TEST M/M/c(Source) ==============================================")
    m = SourceMMc(c=2)
    m.set_data(M=5, rate=1.0/30.0, mu=1.0/3.0)
    print(f"Offered load is {m.get_offered_load()}, Answer is 0.1")
    print(f"P0 = {m.get_P_n(0)}, Answer is 0.619")
    print(f"System length = {m.get_system_length()}, Answer is 0.465")
    print(f"System wait time = {m.get_system_wait_time()}, Answer is 3.075")


test_MM1()
test_MMc()
test_MMcK()
test_MMcc()
test_infinite_capacity()
test_sourceMMc()