def nchoosek(n, k):
    if k == 0:
        return 1
    return 1.0 * n * nchoosek(n-1, k-1) / k
    

class HyperGeometric:
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
        
    def cdf(self, sample_num, false_sample_num):
        if false_sample_num > self.total_false_num or sample_num > self.total_num:
            return None
        
        min_false_num = max(0, self.total_false_num + sample_num - self.total_num)
        max_false_num = min(false_sample_num, min(self.total_false_num, sample_num))
        prob = 0.0
        for i in range(min_false_num, max_false_num + 1):
            prob += self.pdf(sample_num, i)
            print(i, self.pdf(sample_num, i))
        return prob

class OCCurveEng:
    def __init__(total_num):
        self.total_num = total_num
        
    def print_oc_curve(self, n, c):
        X = HyperGeometric(self.total_num, 0)
        for i range()
        
