class Binomial:
    def __init__(self, n, p):
        """
        Initizlize binomial distribution and gen binomial's pdf/cdf.
        
        Args:
            p: float, positive outcome probability in a random trial, 0.0 <= p <= 1.0
            n: int, trial number, n > 0
        """
        import numpy as np
        self.n = n
        self.p = p
        self.var = np.arange(n + 1)
        self.pdf = np.zeros(n + 1)
        self.cdf = np.zeros(n + 1)
        self._gen_binomial()
        
    def _gen_binomial(self):
        numerator = 1.0
        denominator = 1.0
        base_prob = (1-self.p) ** (self.n)
        scale = self.p / (1 - self.p)
        
        self.pdf[0] = base_prob
        self.cdf[0] = base_prob
        for i in range(0, self.n):
            numerator *= (self.n - i)
            denominator *= (i + 1.0)
            base_prob *= scale
            self.pdf[i+1] = numerator * base_prob / denominator
            self.cdf[i+1] = self.cdf[i] + self.pdf[i + 1]

    def prob(self, x):
        if x < 0 or x > self.n:
            print("error input")
            return None
        else:
            return self.pdf[x]
    
    def cumul_prob(self, x):
        if x < 0 or x > self.n:
            print("error input")
            return None
        else:
            return self.cdf[x]
