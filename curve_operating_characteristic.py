import matplotlib.pyplot as plt    
import numpy as np


def nchoosek(n, k):
    if k == 0:
        return 1.0
    return 1.0 * n * nchoosek(n-1, k-1) / k
    

class HyperGeometric:
    """
    gen a HyperGeometric random variable
    """
    def __init__(self, total_num, total_false_num):
        if total_num < total_false_num:
            return
        
        self.total_num = total_num
        self.total_false_num = total_false_num
        self.total_true_num = total_num - total_false_num
    
    def reset_total_false_num(self, total_false_num):
        if total_false_num > self.total_num:
            return
        self.total_false_num = total_false_num
        self.total_true_num = self.total_num - total_false_num
    
    def pdf(self, sample_num, false_sample_num):
        true_sample_num = sample_num - false_sample_num
        false_cnt = nchoosek(self.total_false_num, false_sample_num)
        true_cnt = nchoosek(self.total_true_num, true_sample_num)
        total_cnt = nchoosek(self.total_num, sample_num) 
        return false_cnt * true_cnt / total_cnt
        
    def cdf(self, sample_num, threshold):
        """
        Compute P(X <= threshold) where X is HyperGeometric Distribution with:
            1. total unit number = self.total_num
            2. total false number = self.total_false_num
            3. total sample number = sample_num
            
        Args:
            sample_num(int): sample number of X
            threshold(int): the upper bound of the X in computing cdf
            
        Returns:
            prob(float): cumulative probability which equals to P(X <= threshold)
        """
        if threshold > sample_num or sample_num > self.total_num:
            return None
        
        min_false_num = max(0, self.total_false_num + sample_num - self.total_num)
        max_false_num = min(threshold, self.total_false_num, sample_num)
        prob = 0.0
        #print("\n", threshold, self.total_false_num, sample_num)
        for i in range(min_false_num, max_false_num + 1):
            prob += self.pdf(sample_num, i)
            #print(i, self.pdf(sample_num, i))
        #print(prob)
        return prob

class OCCurve:
    """
    a tool to gen operating characteristic curve.
    """
    def __init__(self, total_num):
        self.total_num = total_num
        
    def print_oc_curve(self, n, c, opt = '.r'):
        hg = HyperGeometric(self.total_num, 0)
        X0 = np.arange(0, 0.5, 0.01)
        Y0 = np.zeros(len(X0))
        
        for i in range(len(X0)):
            hg.reset_total_false_num(int(round(X0[i] * self.total_num)))
            Y0[i] = hg.cdf(n, c)
            
        plt.plot(X0, Y0, opt)
    
    def gen_oc_curve_dict(self, sample_num, acpt_num, opt = '.r'):
        return {'sample_num': sample_num, 'acpt_num': acpt_num, 'opt': opt}
        
    def series_oc_curve1(self, sample_num):
        curve_lt = []
        curve_lt.append(self.gen_oc_curve_dict(sample_num, 1, '.r'))
        curve_lt.append(self.gen_oc_curve_dict(sample_num, 2, '.b'))
        curve_lt.append(self.gen_oc_curve_dict(sample_num, 3, '.g'))
        curve_lt.append(self.gen_oc_curve_dict(sample_num, 4, '.y'))
        curve_lt.append(self.gen_oc_curve_dict(sample_num, 5, '.k'))
        
        for i in range(len(curve_lt)):
             self.print_oc_curve(curve_lt[i]['sample_num'], curve_lt[i]['acpt_num'], curve_lt[i]['opt'])
        
    def series_oc_curve2(self, acpt_num):
        curve_lt = []
        curve_lt.append(self.gen_oc_curve_dict(10, acpt_num, '.r'))
        curve_lt.append(self.gen_oc_curve_dict(30, acpt_num, '.b'))
        curve_lt.append(self.gen_oc_curve_dict(50, acpt_num, '.g'))
        curve_lt.append(self.gen_oc_curve_dict(70, acpt_num, '.y'))
        curve_lt.append(self.gen_oc_curve_dict(90, acpt_num, '.k'))
        
        for i in range(len(curve_lt)):
             self.print_oc_curve(curve_lt[i]['sample_num'], curve_lt[i]['acpt_num'], curve_lt[i]['opt'])
 
def test1(num):
    t1 = OCCurve(num)  
    t1.series_oc_curve1(50)
    
