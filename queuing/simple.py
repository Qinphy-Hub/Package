import math


class Model(object):
    def __init__(self, c: int=None, K: int=None):
        if c is not None and c < 1:
            raise ValueError("The number of services can not be less than 1.")
        if K is not None and K < 1:
            raise ValueError("The capacity of the system not be less than 1.")
        if c is None and K is not None:
            raise ValueError("Not support M/M/infinite/K/FCFS model.")
        self.__c = c
        self.__K = K
        """ simple Markov queueing models

        :param c    int(default=None, means infinite)   the number of services

        :param K    int(default=None, means infinite)   the capacity of system
        """
        self.__L_q = None

    def description(self):
        c = "infinite" if self.__c is None else self.__c
        K = "infinite" if self.__K is None else self.__K
        print(f"Model: M/M/{c}/{K}/FCFS")

    def set_data(self, rate, mu):
        if rate < 0:
            raise ValueError("Arrival rate should not be less than 0.")
        if mu < 0:
            raise ValueError("Service time should not be less than 0.")
        self.__lambda = rate
        self.__lambda_eff = rate
        self.__mu = mu
        self.__rho = None if self.__c is None else rate / (self.__c * mu)
        self.__r = rate / mu
        self.__L_q = self.__calc_L_q()
        if self.__c is not None and self.__K is not None:
            self.__lambda_eff = self.__lambda * (1 - self.get_P_n(self.__K))
            pass

    # lambda_eff
    def get_effective_arrival_rate(self):
        return self.__lambda_eff

    # rho
    def get_traffic_intensity(self):
        return self.__rho

    # r
    def get_offered_load(self):
        return self.__r

    # Lq
    def get_queue_length(self):
        if self.__L_q is None:
            raise ValueError("Please set data first! Model.set_data(lambda, mu).")
        return self.__L_q

    # Ls
    def get_system_length(self):
        return self.get_queue_length() + self.__lambda_eff / self.__mu

    # Wq
    def get_queue_wait_time(self):
        return self.get_system_wait_time() - 1.0 / self.__mu

    # Ws
    def get_system_wait_time(self):
        return self.get_system_length() / self.__lambda_eff

    def __calc_L_q(self):
        if self.__c is not None and self.__K is None:
            # M/M/c/infinite/FCFS
            return self.__MMc_L_q()
        elif self.__c is not None and self.__K is not None:
            # M/M/c/K/FCFS
            return self.__MMcK_L_q()
        elif self.__c is None and self.__K is None:
            # M/M/infinite/infinite/FCFS
            return 0

    def get_P_n(self, n: int):
        if n < 0:
            raise ValueError("n can not be less than 0 in Pn.")
        if self.__c is not None and self.__K is None:
            # M/M/c/infinite/FCFS
            return self.__MMc_P_n(n)
        elif self.__c is not None and self.__K is not None:
            # M/M/c/K/FCFS
            return self.__MMcK_P_n(n)
        elif self.__c is None and self.__K is None:
            # M/M/infinite/infinite/FCFS
            return self.__MMinfty_P_n(n)

    def get_probability_of_queue_wait_time(self, t: float):
        if t < 0:
            raise ValueError("Time can not be less than 0.")
        if self.__c is not None and self.__K is None:
            # M/M/c/infinite/FCFS
            return self.__MMc_W_q_t(t)
        elif self.__c is not None and self.__K is not None:
            # M/M/c/K/FCFS
            return self.__MMcK_W_q_t(t)
        elif self.__c is None and self.__K is None:
            # M/M/infinite/infinite/FCFS
            return 1

    def ErlangB(self, c=None, r=None):
        if self.__c is not None and self.__K is not None and self.__c == self.__K:
            c = self.__c if c is None else c
            r = self.__r if r is None else r
            A = (r ** c) / math.factorial(c)
            B = 0
            for i in range(c + 1):
                B += ((r ** i) / math.factorial(i))
            return A / B
        else:
            raise ValueError("Erlang B formula just for M/M/c/c model.")

    def ErlangC(self, c=None, r=None):
        if self.__c is not None and self.__K is None:
            c = self.__c if c is None else c
            r = self.__r if r is None else r
            rho = r / c
            A = (r ** c) / (math.factorial(c) * (1 - rho))
            B = (r ** c) / (math.factorial(c) * (1 - rho))
            for i in range(c):
                B += ((r ** i) / math.factorial(i))
            return A / B
        else:
            raise ValueError("Erlang C formula just for M/M/c model.")

    def busy_period_analysis(self):
        """ Busy period analysis
        :return time        the time length of busy period
        :return loop_time   the time length between adjacent busy period
        """
        if self.__c is not None and self.__K is None:
            time = 1.0 / (self.__mu - self.__lambda)
            loop_time = 1.0 / self.__lambda + 1.0 / (self.__mu - self.__lambda)
            return time, loop_time
        else:
            raise ValueError("Busy period analysis only for M/M/c model.")

    # ========================================== M/M/c ==========================================
    def __MMc_P_n(self, n):
        if n == 0:
            A = (self.__r ** self.__c) / (math.factorial(self.__c) * (1 - self.__rho))
            B = 0
            for i in range(self.__c):
                B += ((self.__r ** i) / math.factorial(i))
            return 1.0 / (A + B)
        elif 0 < n < self.__c:
            A = math.factorial(n) * (self.__mu ** n)
            return (self.__lambda ** n) / A * self.__MMc_P_n(0)
        else:
            A = (self.__c ** (n - self.__c)) * math.factorial(self.__c) * (self.__mu ** n)
            return (self.__lambda ** n) / A * self.__MMc_P_n(0)

    def __MMc_L_q(self):
        B = math.factorial(self.__c) * ((1 - self.__rho) ** 2)
        return (self.__r ** self.__c) * self.__rho / B * self.__MMc_P_n(0)

    def __MMc_W_q_t(self, t):
        A = (self.__r ** self.__c) * self.__MMc_P_n(0)
        B = math.factorial(self.__c) * (1 - self.__rho)
        C = math.exp(-(self.__c * self.__mu - self.__lambda) * t)
        return 1 - A / B * C


    # ========================================= M/M/c/K =========================================
    def __MMcK_P_n(self, n):
        if n == 0 and self.__rho == 1:
            A = (self.__r ** self.__c) * (self.__K - self.__c + 1)
            for i in range(self.X):
                A += ((self.__r ** i) / math.factorial(i))
            return 1.0 / A
        elif n == 0:
            A = 1 - (self.__rho ** (self.__K - self.__c + 1))
            B = (self.__r ** self.__c) / math.factorial(self.__c) * A / (1 - self.__rho)
            for i in range(self.__c):
                B += ((self.__r ** i) / math.factorial(i))
            return 1.0 / B
        elif n < self.__c:
            A = math.factorial(n) * (self.__mu ** n)
            return (self.__lambda ** n) / A * self.__MMcK_P_n(0)
        elif n <= self.__K:
            A = math.factorial(self.__c) * (self.__c ** (n - self.__c)) * (self.__mu ** n)
            return (self.__lambda ** n) / A * self.__MMcK_P_n(0)
        else:
            raise ValueError("In M/M/c/K/FCFS model: n <= K(capacity of the system).")

    def __MMcK_L_q(self):
        if self.__rho != 1:
            A = self.__MMcK_P_n(0) * (self.__r ** self.__c) * self.__rho
            B = math.factorial(self.__c) * ((1 - self.__rho) ** 2)
            c1 = self.__rho ** (self.__K - self.__c + 1)
            c2 = (1 - self.__rho) * (self.__K - self.__c + 1) * (self.__rho ** (self.__K - self.__c))
            C = 1 - c1 - c2
            return A / B * C
        else:
            A = (self.__c ** self.__c) / math.factorial(self.__c)
            B = (self.__K - self.__c) * (self.__K - self.__c + 1) / 2
            C = 0
            for i in range(self.__c):
                c1 = (self.__c ** i) / math.factorial(i)
                c2 = (self.__c ** self.__c) / math.factorial(self.__c) * (self.__K - self.__c + 1)
                C += (c1 + c2)
            return (A * B) / C

    def __MMcKD_Q_n(self, n):
        if n <= self.__K - 1:
            return self.__MMcK_P_n(n) / (1 - self.__MMcK_P_n(self.__K))
        else:
            raise ValueError("In M/M/c/K/FCFS model: n <= K(capacity of the system).")

    def __MMcK_W_q_t(self, t):
        A = 0
        for i in range(self.__c, self.__K, 1):
            B = 0
            for j in range(i - self.__c + 1):
                b1 = (self.__c * self.__mu * t) ** j
                b2 = math.exp(-self.__c * self.__mu * t)
                B += (b1 * b2 / math.factorial(j))
            A += (self.__MMcKD_Q_n(i) * B)
        return 1 - A

    # ===================================== M/M/infinite =====================================
    def __MMinfty_P_n(self, n):
        return (self.__r ** n) * math.exp(-self.__r) / math.factorial(n)



class SourceMMc(object):
    def __init__(self, c: int):
        if c < 1:
            raise ValueError("The number of services can not be less than 1.")
        self.__c = c
        self.__L_s = None

    def set_data(self, M: int, rate: float, mu: float):
        if rate < 0:
            raise ValueError("Arrival rate should not be less than 0.")
        if mu < 0:
            raise ValueError("Service time should not be less than 0.")
        if M < self.__c:
            raise ValueError("The size of the finite customer source is no less than c.")
        self.__M = M
        self.__lambda = rate
        self.__mu = mu
        self.__rho = rate / (self.__c * mu)
        self.__r = rate / mu
        self.__coeff = self.__calc_P_coeff()
        self.__L_s = self.__calc_L_s()
        self.__lambda_eff = self.__calc_lambda_eff()

    def __calc_P_coeff(self):
        a = [1]
        for i in range(1, self.__c, 1):
            a.append(math.comb(self.__M, i) * (self.__r ** i))
        for i in range(self.__c, self.__M + 1, 1):
            a0 = math.comb(self.__M, i)
            a1 = math.factorial(i) * (self.__r ** i)
            a2 = (self.__c ** (i - self.__c)) * math.factorial(self.__c)
            a.append(a0 * a1 / a2)
        return a

    # Ls
    def __calc_L_s(self):
        p0 = self.get_P_n(0)
        A = 0
        for i in range(1, self.__M + 1, 1):
            A += (i * self.__coeff[i])
        return p0 * A

    # lambda_eff
    def __calc_lambda_eff(self):
        return self.__lambda * (self.__M - self.__L_s)

    # rho
    def get_traffic_intensity(self):
        return self.__rho

    # r
    def get_offered_load(self):
        return self.__r

    # lambda_eff
    def get_effective_arrival_rate(self):
        return self.__lambda_eff

    # Pn
    def get_P_n(self, n: int):
        if n < 0 or n > self.__M:
            raise ValueError("n can not be less than 0 and more than M in Pn.")
        p0 = 1.0 / sum(self.__coeff)
        return self.__coeff[n] * p0

    # Ls
    def get_system_length(self):
        if self.__L_s is None:
            raise ValueError("Please set data first! Model.set_data(M, lambda, mu).")
        return self.__L_s

    # Lq
    def get_queue_length(self):
        return self.__L_s - self.__r * (self.__M - self.__L_s)

    # Ws
    def get_system_wait_time(self):
        return self.__L_s / self.__lambda_eff

    # Wq
    def get_queue_wait_time(self):
        return self.get_queue_length() / self.__lambda_eff

    
        


        
        
        


        