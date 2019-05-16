# Operating Characteristic Curves - Acceptance Sampling for Attributes

This procedure is used view graphically <font color="blue">the probability of
lot acceptance</font> versus <font color="blue">the lot proportion defective </font> <font color="red">for a given sample size and acceptance number</font>.

換句話說，也就是給定一個抽樣方法(固定樣本數及樣本最大瑕疵數)，在不同實際的defect比例下，整批資料被accept的機率為多少。

這是品質管理下，常見用來挑選適當抽樣方法的一種圖像化表示。

---

## 變數定義

- N：母體的unit數
- p：母體中，瑕疵unit佔的比例 (p = K / N)
- K：母體中，瑕疵unit的個數 (K = N * p)
- n：樣本數
- c：某個抽樣方法允許的最大容錯數
- Pa (Probability of Acceptance)：給定特定母體(N,K)下，並滿足特定抽樣條件(n,c)的機率：
    - Pa = P(X <= c), where X ~ 某個分配
        - X ~ Hypergeometric(N, K, n)：通常母體N數量不大的情況下，給定抽取樣本數n，則X滿足超幾何分配。
        - X ~ Binomial(n, p)：若母體N總數很大且 n << N，上式近似為二項式分布。

## Operating Characteristic Curves

![](https://i.imgur.com/EXMx6YP.png)

圖中不同顏色的曲線，各自代表一個抽樣方法被acceptance的機率，其中x軸為假設實際的瑕疵率為p的話。而曲線表示一種特定抽樣標準，定義為：

&emsp;&emsp;從母體N中，抽取n個樣本，並限定樣本中最大瑕疵數不得超過c。當實際的瑕疵數K(或瑕疵比例p)為某個值時，這個抽樣方法會被acceptance的機率。

若以圖中藍線為例子，則：
- N = 500
- n = 50
- c = 2


可以做一些簡單的觀察：
- 若實際的瑕疵數K = 0，也就是 p = 0.0，不論是哪一種抽樣方法，都不可能抽到任何瑕疵品，因此當母體是毫無瑕疵品時，各種抽樣方法都100%會被acceptance。
- 若實際的p = 0.05，圖的紅色曲線，代表50個樣本都不能有瑕疵品，這件事情發生的機率，只有不到10%；反之藍色曲線代表50個樣本最多容許2個瑕疵品，這件事情發生的機率有將近50%。

反過來說，若我們希望在圖中選擇一個抽樣方式(曲線)，可以acceptance下列敘述：當實際母體的瑕疵率小於5%，該抽樣方式要有70%以上的機率挑出這種母體(下圖有粗紅點的線都符合條件)，則最終可能會選到橘色的線 ── 50個樣本中瑕疵品不能超過3個。

![](https://i.imgur.com/a9C0fm9.png)

### Code
```python
import matplotlib.pyplot as plt    
import numpy as np

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


def find_curve_test(population_num = 500, sample_num = 50):
    oc_curve_eng = OCCurveEngine(population_num)
    limit_lt = [0,1,2,3,4,5]
    color_lt = ['red', 'green', 'blue', 'coral', 'gray', 'purple']
    for i in range(len(color_lt)):
        curve = oc_curve_eng.get_oc_curve(sample_num, limit_lt[i])
        oc_curve_eng.plot_oc_curve(curve, color = color_lt[i])
    
    hg = HyperGeometric(population_num, int(round(0.05 * population_num)), sample_num)        
    for i in range(len(limit_lt)):
        y = hg.cdf(limit_lt[i])
        if y > 0.7: #Probability of Acceptance over than 70%
            plt.plot(0.05, y, '*r')
        else:
            plt.plot(0.05, y, '*k')
    
    plt.grid()
    plt.show() 
```


## HyperGeometric

### Definition

The following conditions characterize the hypergeometric distribution:

- The result of each draw (the elements of the population being sampled) can be classified into one of two mutually exclusive categories (e.g. Pass/Fail or Employed/Unemployed).
- The probability of a success changes on each draw, as each draw decreases the population (sampling without replacement from a finite population).

A random variable **X** follows the hypergeometric distribution if its probability mass function (pmf) is given by

&emsp;&emsp;![](https://i.imgur.com/xBEOaw3.png)

where
![](https://i.imgur.com/Q0JrPit.png)

A random variable distributed hypergeometrically with parameters **N**, **K** and **n** is written：

&emsp;&emsp;![](https://i.imgur.com/a86OOV0.png)


### Code

```python
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
```
