# Simple Random Sampling Without Replacement (SRSWOR)

SRSWOR is a method of selection of n units out of the N units one by one such that at any stage of selection, anyone of the remaining units have same chance of being selected.

The sampling units are chosen without replacement. So, the units once chosen are not placed back in the population.

----
Reference:
- [Simple Random Sample](http://home.iitk.ac.in/~shalab/sampling/chapter2-sampling-simple-random-sampling.pdf)
----

## Notation

![](https://i.imgur.com/0ZRKMJk.png)


## Estimation of population mean

從母體的N個單元中，不重複抽取n個單元，共有$C^N_n$種組合，因此每一種組合發生的機率為$\frac{1}{C^N_n}$

假設 $t_i$ 代表第i組樣本平均數 $\overline{y}_i$，則$E(\overline{y})$ 為 ${C^N_n}$ 個t的平均數。

![](https://i.imgur.com/DjlrZ5j.png)

從N個母體當中不重複抽出n個樣本，形成的所有組合中。 任何一個母體單元出現的個數為$C^{N-1}_{n - 1}$ (排列組合：包含某特定樣本的所有組合)。也可以換個角度想，兩個summatio展開後，出現y的總個數為${C^N_n} * n$，我們又知道每個y的出現次數應該相同，因此平均分給N個樣本則得到${C^N_n} * n / N = C^{N-1}_{n - 1}$。

有了樣本平均跟母體平均的關聯後，便能求得樣本平均的期望值剛好也是母體平均，因此樣本平均同時也是母體平均的不偏估計量。

![](https://i.imgur.com/lWsctV1.png)

![](https://i.imgur.com/3eBgLKv.png)


## Variance of the estimate

![](https://i.imgur.com/tDKdjfM.png)

樣本平均的變異數根據定義展開後，最終得到S跟K的函式。

- 大寫S是用母體算得的，並不是傳統的樣本變異數
- K中summation的上標都是小寫n，需要做一些換算取得跟大寫N的關聯。

![](https://i.imgur.com/DlDTaL5.png)

有了K跟S的關聯後

併入樣本平均的變異數中，求得：

![](https://i.imgur.com/BdSpvPr.png)

- 再重申，S為母體的參數變換而來的，真實情境中並不知道母體，所以欲求得樣本平均的變異數，還需要對S再做估計。

## Estimation of variance from a sample

對參數的估計有很多種形式，較為常見的就是點估計及最大概似估計，不像最大概似估計有一套很明確的流程，點估計並沒有一定的方法，很多時候都憑藉著經驗和猜測得到。

這裡就是用猜的，通常在猜測這種估計值時，都有一些依據。

假設我們都認同當母體N非常大的話，正常的樣本平均變異數 s^2 的公式如下面所述。
當我們對 s^2 取期望值，若能發現跟 S^2 有某種關聯，我們就能將 s^2 做一些轉換後，估計出S^2。

![](https://i.imgur.com/o0hlnGv.png)

### In case of SRSWOR

![](https://i.imgur.com/ICqiUDs.png)

最終得到 E(s^2) = S^2 ，換句話說 s^2 剛好就是 S^2 的不偏估計量

再代回Variance of the estimate中推演過的樣本變異數中，便得到：

![](https://i.imgur.com/GrAdFX6.png)


## Test Code

在SRSWOR中，真實的樣本平均變異數衰退的很快，主要是因為(N-n)/(Nn)衰退的速度很快，我們也能發現，當N = n的時候，樣本平均就是母體平均，樣本平均變異數為0也是符合預期的。

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_decline_variance_in_srswor(sample_variance, total_num):
    """
    Plot the decline of variance when the number of sample increase in the SRSWOR case.
    
    sample_variance(float): sample variance in all sample number case
    total_num(int): number of population
    """
    sample_num_npary = np.arange(5, total_num, 5)
    real_variance_npary = np.zeros(len(sample_num_npary))
    for i in range(len(sample_num_npary)):
        real_variance_npary[i] = sample_variance * (total_num - sample_num_npary[i]) / (total_num * sample_num_npary[i])
        
    plt.plot(sample_num_npary, real_variance_npary, '.r')
```

下圖為假設不論樣本多少，s^2 都是10，且母體總數為100。
在SRSWOR情境下的實際樣本平均變異數為。

![](https://i.imgur.com/8bjQpcs.png)

x軸代表取樣的數量，間隔為5，大約1成的樣本數就足以讓真實的變異數衰退到傳統變異數的一成以下了。