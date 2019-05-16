# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt    
import numpy as np


def nchoosek(n, k):
    if k == 0:
        return 1.0
    return 1.0 * n * nchoosek(n-1, k-1) / k
    

class HyperGeometric:
    def __init__(self, N, K, n):
        self.N = N
        self.K = K
        self.n = n
        self.L = N - K
        self.lower_limit = max(0, n - self.L)
        self.upper_limit = min(n, K)
    
    
    def __new__(cls,  N, K, n):
        if N < K or N < n:
            return None        
        else:
            return object.__new__(cls)
    
    
    def reset_parameters(self, N, K, n):
        if N < K or N < n:
            return False        
        self.N = N
        self.K = K
        self.n = n
        self.L = N - K
        self.lower_limit = max(0, n - self.L)
        self.upper_limit = min(n, K)
        return True
    
    def pdf(self, unit_num):
        if unit_num > self.upper_limit or unit_num < self.lower_limit:
            return 0.0
        
        K_cnt = nchoosek(self.K, unit_num)
        L_cnt = nchoosek(self.L, self.n - unit_num)
        total_cnt = nchoosek(self.N, self.n) 
        return K_cnt * L_cnt / total_cnt

        
    def cdf(self, threshold):
        """
        Compute P(X <= threshold) where X ~ HyperGeometric(N,K,n) :
            1. N = self.N
            2. K = self.K
            3. n = self.n
            
        Args:
            threshold(int): a limit that X can't greater than the threshold 
            
        Returns:
            prob(float): cumulative probability: P(X <= threshold)
        """
        if threshold >= self.upper_limit:
            return 1.0
        
        if threshold < self.lower_limit:
            return 0.0

        prob = 0.0
        for i in range(self.lower_limit, threshold + 1):
            prob += self.pdf(i)

        return prob


class OCCurveEngine:
    def __init__(self, population_num):
        self.population_num = population_num
        
    def get_oc_curve(self, sample_num, limit_num):
        hg = HyperGeometric(self.population_num, 0, sample_num)
        if hg == None:
            return None
        x = np.arange(0, 0.5, 0.01)
        y = np.zeros(len(x))
        
        for i in range(len(x)):
            hg.reset_parameters(self.population_num, int(round(x[i] * self.population_num)), sample_num)
            y[i] = hg.cdf(limit_num)
        
        return {'population_num':self.population_num, 'sample_num':sample_num, 'limit_num':limit_num, 'defect_proportion':x, 'probability_of_acceptance':y}
        
    def plot_oc_curve(self, oc_curve, color = 'red', linestyle = ':'):
        if oc_curve == None:
            return
        plt.plot(oc_curve['defect_proportion'], oc_curve['probability_of_acceptance'], color = color, linestyle = linestyle)
        
        
def show_task1(population_num = 500, sample_num = 50):
    oc_curve_eng = OCCurveEngine(population_num)
    limit_lt = [0,1,2,3,4,5]
    color_lt = ['red', 'green', 'blue', 'coral', 'gray', 'purple']
    for i in range(len(color_lt)):
        curve = oc_curve_eng.get_oc_curve(sample_num, limit_lt[i])
        oc_curve_eng.plot_oc_curve(curve, color = color_lt[i])
    
    hg = HyperGeometric(population_num, int(round(0.05 * population_num)), sample_num)        
    for i in range(len(limit_lt)):
        y = hg.cdf(limit_lt[i])
        if y > 0.6:
            plt.plot(0.05, y, '*r')
        else:
            plt.plot(0.05, y, '*k')
    
    plt.grid()
    plt.show()    
